# Animation Pin Tool
# -------------------------------------------------------------------- #

__author__  = "Daniel Klug"
__version__ = "1.50"
__date__    = "10-15-2021"
__email__   = "daniel@redforty.com"

# -------------------------------------------------------------------- #
'''
Installation:
Place both files in your maya/scripts folder, then restart Maya

Usage:
To launch the UI, use this python command:
import animPin; animPin.UI()

By command line from within the script editor:
import animPin as animPin; animPin.create_pins() # To create pins
import animPin as animPin; animPin.bake_pins() # To bake selected pin groups

'''
# Imports ============================================================ #

import maya.cmds as cmds
import maya.mel as mel
import maya.api.OpenMaya as api
import maya.api.OpenMayaAnim as anim
import maya.OpenMayaUI as mui

# Qt is a project by Marcus Ottosson-> https://github.com/mottosso/Qt.py
from Qt import QtGui, QtCore, QtCompat, QtWidgets, __binding__
# from Qt.QtGui import QPen, QColor, QBrush, QLinearGradient

if "PyQt" == __binding__:
    import sip
elif "PySide" == __binding__:
    import shiboken as shiboken  # Do Pyside
elif "PySide2" == __binding__:
    import shiboken2 as shiboken # You're on Maya 2018, aren't you?

import time
import base64
import itertools
from collections import OrderedDict
from functools import wraps
import re # I'm so sorry

# Globals ============================================================ #

WIDTH = 180
HEIGHT = 462
_view = None
ap_suffix= '_pin'
pin_group = 'pin_group#'
master_group = 'animPins_group'
locator_scale = 10 # Season to taste - depends on your rig size
pin_data = OrderedDict([
    ('control'               , {'dataType'     : "string"}),
    ('constraint'            , {'dataType'     : "string"}),
    ('start_frame'           , {'attributeType':  "float"}),
    ('end_frame'             , {'attributeType':  "float"}),
    # ('translates_enabled'    , {'attributeType':   "bool"}), # We can use the locked channels
    # ('rotates_enabled'       , {'attributeType':   "bool"}),
    ('translate_keys'        , {'dataType'     : "string"}),
    ('rotate_keys'           , {'dataType'     : "string"}),
    ('translate_locked'      , {'attributeType':   "bool"}),
    ('rotate_locked'         , {'attributeType':   "bool"}),
    ('preserve_blendParent'  , {'dataType'     : "string"}) # Was there a blendparent node there to begin with?
])
# Get the timeline object
aPlayBackSliderPython = mel.eval('$tmpVar=$gPlayBackSlider')
tuple_rex = re.compile("([0-9]+\.[0-9]+\, [0-9]+\.[0-9]+)")

# Developer section ================================================== #
if __name__ == '__main__':
    try:
        AnimPinUI.stop() #pylint:disable=E0601
    except:
        pass

# Decorators ========================================================= #

def viewport_off(func):
    """
    Decorator - turn off Maya display while func is running. if func fails, the error will be raised after.
    """
    @wraps(func)
    def wrap( *args, **kwargs ):

        parallel = False
        if 'parallel' in cmds.evaluationManager(q=True, mode=True):
            cmds.evaluationManager(mode='off')
            parallel = True
            print "Turning off Parallel evaluation..."
        # Turn $gMainPane Off:
        mel.eval("paneLayout -e -manage false $gMainPane")
        cmds.refresh(suspend=True)
        # Hide the timeslider
        mel.eval("setTimeSliderVisible 0;")


        # Decorator will try/except running the function.
        # But it will always turn on the viewport at the end.
        # In case the function failed, it will prevent leaving maya viewport off.
        try:
            return func( *args, **kwargs )
        except Exception:
            raise # will raise original error
        finally:
            cmds.refresh(suspend=False)
            mel.eval("setTimeSliderVisible 1;")
            if parallel:
                cmds.evaluationManager(mode='parallel')
                print "Turning on Parallel evaluation..."
            mel.eval("paneLayout -e -manage true $gMainPane")
            cmds.refresh()

    return wrap


def undo(func):
    '''
    Decorator - open/close undo chunk
    '''
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(openChunk = True)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise # will raise original error
        finally:
            cmds.undoInfo(closeChunk = True)
            # cmds.undo()

    return wrap


def noUndo(func):
    '''
    Decorator - open/close undo chunk
    '''
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(stateWithoutFlush = False)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise # will raise original error
        finally:
            cmds.undoInfo(stateWithoutFlush = True)
            # cmds.undo()

    return wrap

# Public methods ===================================================== #

@undo
def create_pins(selection = None, start_frame = None, end_frame = None, group_override = None):
    '''
    selection: could be a string, list of strings, or MSelectionList
    start_frame: float
    end_frame: float
    group_override = None
    '''
    # Validate input ------------------------------------------------- #

    current_state = None
    if cmds.optionVar(exists='animBlendingOpt'):
        current_state = cmds.optionVar(query='animBlendingOpt')
    if not current_state:
        cmds.error('Animation blending preference is NOT SET!')
        return None

    sel_list = _get_selectionList(selection)
    controls = _validate_selection(sel_list)
    if controls.isEmpty():
        api.MGlobal.displayError(\
            "Could not find any valid nodes to pin.")
        return None

    start_frame, end_frame = _validate_framerange(start_frame, end_frame)
    if start_frame == None or end_frame == None:
        api.MGlobal.displayError(\
            "Could not validate frame range from %s to %s." % \
            (start_frame, end_frame))
        return None

    new_pin_group = _create_new_pin_group()
    if not new_pin_group:
        api.MGlobal.displayError(\
            "Could not create a valid group to add pins.")
        return None

    # Begin alpha setup ---------------------------------------------- #

    locators = []
    constraints = []
    for i in range(controls.length()):
        # This is the part that makes this tool special
        control = controls.getDependNode(i)
        control_name = api.MFnDependencyNode(control).name()
        control_data = _read_control_data(control, start_frame, end_frame)
        locator = _create_locator_pin(control_data, new_pin_group)
        locators.append(locator)
        constraint_node = cmds.parentConstraint(control_name, locator)[0]
        constraints.append(constraint_node)
        cmds.setKeyframe(locator)
        cmds.setAttr(locator + '.blendParent1', 1)

    # Do magic ------------------------------------------------------- #
    results = _do_bake(locators, start_frame, end_frame)
    if not results:
        raise ValueError("Bake failed.")
    cmds.delete(constraints)

    # Begin omega setup ---------------------------------------------- #
    pins = _get_pins(new_pin_group)
    for pin in pins:
        control = cmds.getAttr(pin + '.control')
        MSel = api.MGlobal.getSelectionListByName(control)
        controlFN = api.MFnDependencyNode(MSel.getDependNode(0))
        if _is_locked_or_not_keyable(controlFN, 'translate'):
            skip_translate = ['x', 'y', 'z']
        else:
            skip_translate = 'none'
        if _is_locked_or_not_keyable(controlFN, 'rotate'):
            skip_rotate = ['x', 'y', 'z']
        else:
            skip_rotate = 'none'

        constraint_node = cmds.parentConstraint( \
                               pin, \
                               control, \
                               skipTranslate = skip_translate, \
                               skipRotate = skip_rotate)[0] # Reversed
        cmds.setAttr(pin + '.constraint', \
                     constraint_node, \
                     type = 'string')

        try:
            cmds.setKeyframe(control, \
                             attribute="blendParent1", \
                             time=[start_frame, end_frame], \
                             value=1)
            cmds.setKeyframe(control, \
                             attribute="blendParent1", \
                             time=[start_frame-1, end_frame+1], \
                             value=0)
        except:
            api.MGlobal.displayWarning(\
                "Could not key the blendParent. " \
                 "Be careful outside the buffer range!")

        # Lock the attributes of the locator that were locked on the control
        t_lock = cmds.getAttr(pin + '.translate_locked')
        r_lock = cmds.getAttr(pin + '.rotate_locked')
        cmds.setAttr(locator + '.t', lock=t_lock, keyable=(not t_lock))
        cmds.setAttr(locator + '.r', lock=r_lock, keyable=(not r_lock))

    print "Successfully created new pin group, '%s'!" % new_pin_group
    return new_pin_group

@undo
def bake_pins(pin_groups = None, bake_option = 1, start_frame = None, end_frame = None):
    '''
    Bake options
    '''
    if not pin_groups:
        # check selection
        selection = cmds.ls(sl=True)
    pin_groups = _get_pin_groups()
    pin_group_list = set(selection).intersection(pin_groups)
    pins_to_bake = _get_pins(pin_group_list)

    if not pins_to_bake:
        api.MGlobal.displayWarning(\
            "No pins could be found to bake.")
        return None

    constraints = []
    controls_to_bake = []
    blendParents_to_restore = {}
    pin_groups_to_delete = set()
    for pin in pins_to_bake:
        pin_constraint = cmds.getAttr(pin + '.constraint')
        constraints.append(pin_constraint)
        control = cmds.getAttr(pin + '.control')
        if not cmds.objExists(control):
            # Last ditch effort to find the control. Was it renamed?
            control = cmds.listConnections(\
                pin_constraint + '.constraintParentInverseMatrix')[0] \
                or []
            if control: # fix the data
                cmds.setAttr(pin + '.control', control, type = 'string')
        controls_to_bake.append(control)
        pin_parent = cmds.listRelatives(pin, parent=True)[0]
        pin_groups_to_delete.add(pin_parent)
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        blendParents_to_restore[control] = bp_keys

    start_frame, end_frame = _validate_bakerange(\
        pins_to_bake, \
        start_frame, \
        end_frame)
    if start_frame == None or end_frame == None:
        api.MGlobal.displayError(\
            "Could not validate frame range from %s to %s." % \
            (start_frame, end_frame))
        return None

    if bake_option == 0:
        sample = 1
        print "Matching keys..."
    else:
        sample = bake_option
        print "Baking on %ds" % sample

    # Do magic ------------------------------------------------------- #
    success = _do_bake(\
        controls_to_bake, \
        start_frame, \
        end_frame, \
        sample)
    if not success:
        raise ValueError("Bake failed.")
    # cmds.refresh() # Check it out

    for control, bp_keys in blendParents_to_restore.items():
        for match in tuple_rex.finditer(bp_keys):
            cmds.setKeyframe(\
                control, \
                at='blendParent1', \
                time = float(match.group(0).split(', ')[0]), \
                value = float(match.group(0).split(', ')[1]) \
            )

    if bake_option == 0: # Proceed with the Match Keys procedure
        success = _match_keys_procedure(\
            pins_to_bake, \
            start_frame, \
            end_frame)
        if not success:
            raise ValueError("match_keys_procedure failed.")

    cmds.delete(constraints)
    cmds.delete(list(pin_groups_to_delete))
    # Final check to see if the master_group is empty. If so, delete it.
    global master_group
    pins_exist = _get_pins()
    if not pins_exist:
        cmds.delete(master_group)
    print "Successfully baked these pin groups:"
    for pin_group in list(pin_groups_to_delete):
        print pin_group
    return pin_groups_to_delete # We're done here!


# Private methods ==================================================== #

def _to_ranges(iterable):
    # https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python/43091576#43091576
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def _validate_bakerange(pins_to_bake, start_frame, end_frame):
    all_pins_start_frame = set()
    all_pins_end_frame   = set()
    for pin in pins_to_bake:
        pin_start = cmds.getAttr(pin + '.start_frame')
        pin_end = cmds.getAttr(pin + '.end_frame')
        all_pins_start_frame.add(pin_start)
        all_pins_end_frame.add(pin_end)
    all_pins_start_frame = list(all_pins_start_frame)[0]
    all_pins_end_frame   = list(all_pins_end_frame)[0]

    selected_range = str(cmds.timeControl(
                         aPlayBackSliderPython,
                         q=True,
                         range=True)
                        )
    selected_range = [float(x) for x in selected_range.strip('"').split(':')]
    if not selected_range[1] - 1 == selected_range[0]:
        start_frame, end_frame = selected_range

    start_frame, end_frame = _validate_framerange(start_frame, end_frame)

    if start_frame < all_pins_start_frame:
        start_frame = all_pins_start_frame
        # api.MGlobal.displayError(\
        #     "Start frame is before pin start frame. Aborting!")
        # return None

    if end_frame > all_pins_end_frame:
        end_frame = all_pins_end_frame
        # api.MGlobal.displayError(\
        #     "End frame is before pin end frame. Aborting!")
        # return None

    return start_frame, end_frame


def _is_locked_or_not_keyable(controlFN, attribute):
    plug = controlFN.findPlug(attribute, False)
    if any(plug.child(p).isLocked \
        for p in range(plug.numChildren())):
        return True
    if not any(plug.child(p).isKeyable \
        for p in range(plug.numChildren())):
        return True
    for p in range(plug.numChildren()):
        plug_array = plug.child(p).connectedTo(True, False)
        if plug_array:
            if plug_array[0].isChild:
                return True # Assume it is constrained
                # Check to see if it is constrained
                # return plug_array[0].parent().node().hasFn(api.MFn.kParentConstraint):
    return False


def _validate_selection(sel_list):
    current_pins = _get_pins()
    validated_sel_list = api.MSelectionList()

    for i in range(sel_list.length()):
        try:
            dag = sel_list.getDagPath(i)
        except TypeError:
            continue # Quietly skip non-dag nodes

        control_dep = sel_list.getDependNode(i)
        controlFN = api.MFnDependencyNode(control_dep)
        control_name = controlFN.name()

        pinned_control = control_name + ap_suffix
        if pinned_control in current_pins:
            api.MGlobal.displayError(\
                "Node '%s' is already pinned! " \
                "Skipping..." % control_name)
            continue

        if control_name in current_pins:
            api.MGlobal.displayError(\
                "Node '%s' is a pin! " \
                "Skipping..." % control_name)
            continue

        # Had to comment this out. Some controls _are_ joints.
        # if 'animation' in controlFN.classification(controlFN.typeName):
        #     continue # Quietly ignore animation curves

        if not controlFN.typeName in ['transform', 'joint', 'ikHandle']:
            api.MGlobal.displayError(\
                "Node '%s' is not a valid transform node. " \
                "Skipping..." % control_name)
            continue

        if _is_locked_or_not_keyable(controlFN, 'translate') and \
            _is_locked_or_not_keyable(controlFN, 'rotate'):
            api.MGlobal.displayError(\
                "Node '%s' has no available transform channels. " \
                "Skipping..." % control_name)
            continue

        validated_sel_list.add(sel_list.getDependNode(i))

    return validated_sel_list


def _get_pin_groups():
    global master_group
    found_pin_groups = []
    if cmds.objExists(master_group):
        master_group_children = cmds.listRelatives(\
            master_group, \
            type = 'transform') or []
        for found_pin_group in master_group_children:
            found_pin_groups.append(found_pin_group)
    return found_pin_groups # Returns list


def _get_pins(pin_groups = None):
    # search for group
    # ask about override if cant find it
    # If not supplied with a pin group, it gets all pins
    # pin_groups is a string or list of strings
    if isinstance(pin_groups, (str, unicode)): pin_groups = [pin_groups]
    found_pins = []
    if not pin_groups:
        pin_groups = _get_pin_groups()
    for group in pin_groups:
        pins = cmds.listRelatives(group, type = 'transform') or []
        for pin in pins:
            found_pins.append(pin)
    return found_pins


def _validate_framerange(start_frame, end_frame):
    if start_frame == None:
        api.MGlobal.displayWarning(\
            "No start_frame supplied. Defaulting to timeline...")
        start_frame = cmds.playbackOptions(\
            query = True, \
            animationStartTime = True)

    if end_frame == None:
        api.MGlobal.displayWarning(\
            "No end_frame supplied. Defaulting to timeline...")
        end_frame = cmds.playbackOptions(\
            query = True, \
            animationEndTime = True)

    if start_frame > end_frame:
        api.MGlobal.displayError(\
            "Start frame needs to be before end frame!")
        return None

    return start_frame, end_frame


def _read_control_data(control, start_frame, end_frame):
    '''
    control: MObject
    start_frame: float
    end_frame: float
    '''
    # global pin_data # Do we need this
    control_data = OrderedDict()
    controlFN = api.MFnDependencyNode(control)
    control_name = str(controlFN.name())

    t_keys = _get_keys_from_obj_attribute(controlFN, 'translate')
    r_keys = _get_keys_from_obj_attribute(controlFN, 'rotate')

    t_lock = _is_locked_or_not_keyable(controlFN, 'translate')
    r_lock = _is_locked_or_not_keyable(controlFN, 'rotate')

    # Goddammit, I shouldn't be assuming blendparent1
    # Instead, I SHOULD be just tracking which one
    # I created and store the rest. But I'm lazy.
    bp_keys = []
    if 'blendParent1' in cmds.listAttr(control_name):
        bp_key_times = _get_keys_from_obj_attribute(\
            controlFN, \
            'blendParent1')
        if bp_key_times:
            key_values = []
            for time in bp_key_times:
                key_value = cmds.getAttr(\
                    control_name + '.blendParent1', \
                    time = time)
                key_values.append(key_value)
            bp_keys = list(zip(bp_key_times, key_values))
        else:
            bp_keys = [cmds.getAttr(control_name + '.blendParent1')]

    control_data['control'] = control_name
    control_data['start_frame'] = start_frame
    control_data['end_frame'] = end_frame
    control_data['translate_keys'] = t_keys
    control_data['rotate_keys'] = r_keys
    control_data['translate_locked'] = t_lock
    control_data['rotate_locked'] = r_lock
    control_data['preserve_blendParent'] = bp_keys # ugh

    return control_data


def _get_keys_from_obj_attribute(controlFN, attribute):
    keys = set()
    attribute_plug = controlFN.findPlug(attribute, False) # Networked?
    if attribute_plug.isCompound:
        for c in range(attribute_plug.numChildren()):
            plug = attribute_plug.child(c)
            if anim.MAnimUtil.isAnimated(plug):
                keys.update(_get_keys_from_curve(plug))
    else:
        if anim.MAnimUtil.isAnimated(attribute_plug):
            keys.update(_get_keys_from_curve(attribute_plug))
    return list(keys)


def _get_keys_from_curve(plug):
    curve = anim.MFnAnimCurve(plug)
    return [curve.input(k).value for k in range(curve.numKeys)]


def _get_master_group(group_override = None):
    global master_group
    if isinstance(group_override, str):
        master_group = group_override
    if not cmds.objExists(master_group):
        master_group = cmds.createNode('transform',
                                   name = master_group,
                                   skipSelect = True)
    return master_group


def _create_new_pin_group():
    master_group = _get_master_group()
    new_pin_group = cmds.createNode('transform',
                                  name = pin_group,
                                  parent = master_group,
                                  skipSelect = True)
    return new_pin_group


def _get_selectionList(selection):
    if not selection:
        return api.MGlobal.getActiveSelectionList()
    elif isinstance(selection, str):
        return api.MGlobal.getSelectionListByName(selection)
    elif isinstance(selection, list):
        nodes = api.MSelectionList() # Prime the list
        for sel in selection:
            try:
                node = api.MGlobal.getSelectionListByName(sel)
                nodes.merge(node)
            except:
                api.MGlobal.displayError(
                    "Could not fetch selection. " \
                    "Try submitting an MSelectionList " \
                    "or a list of string names."
                )
        return nodes
    else: # You know what you're doing
        return selection


def _create_locator_pin(control_data, pin_group):
    # global pin_data # Do we need this?
    control_name = control_data['control']

    locator = cmds.spaceLocator(name = control_name + ap_suffix)[0]
    cmds.setAttr(locator + ".scale", *[locator_scale]*3) # Unpack * 3
    cmds.parent(locator, pin_group)

    for attr in ['x', 'y', 'z']:
        cmds.setAttr(\
            locator + '.s' + attr, \
            keyable = False, \
            channelBox = True)

    for key, value in pin_data.iteritems():
        cmds.addAttr(\
            locator, \
            longName = key, \
            **value) # kwargs ftw

    for key, value in control_data.items():
        if isinstance(value, (str, list)):
            if isinstance(value, list):
                value = ' '.join(map(str,value))
            cmds.setAttr(locator + '.' + key, value, type = 'string')
        else:
            cmds.setAttr(locator + '.' + key, value)

    return locator


def _get_maya_window():
    ptr = mui.MQtUtil.mainWindow()
    # return QtCompat.wrapInstance(long(ptr), QtWidgets.QMainWindow) # use this when we have QtCompat
    return _wrap_instance(long(ptr), QtWidgets.QMainWindow)


def _wrap_instance(ptr, base=None):
    """
    Utility to convert a pointer to a Qt class instance (PySide/PyQt compatible)

    :param ptr: Pointer to QObject in memory
    :type ptr: long or Swig instance
    :param base: (Optional) Base class to wrap with (Defaults to QObject, which should handle anything)
    :type base: QtGui.QWidget
    :return: QWidget or subclass instance
    :rtype: QtGui.QWidget
    """
    if ptr is None:
        return None
    ptr = long(ptr) #Ensure type
    if globals().has_key('shiboken') and "PySide" in __binding__:
        if base is None:
            qObj = shiboken.wrapInstance(long(ptr), QtCore.QObject)
            metaObj = qObj.metaObject()
            cls = metaObj.className()
            superCls = metaObj.superClass().className()
            if hasattr(QtGui, cls):
                base = getattr(QtGui, cls)
            elif hasattr(QtGui, superCls):
                base = getattr(QtGui, superCls)
            else:
                base = QtGui.QWidget
        return shiboken.wrapInstance(long(ptr), base)
    elif globals().has_key('sip') and "PyQt" in __binding__:
        base = QtCore.QObject
        return sip.wrapinstance(long(ptr), base)
    else:
        # return None
        ptr = mui.MQtUtil.mainWindow()
        return QtCompat.wrapInstance(long(ptr), QtWidgets.QMainWindow)


# Decorated methods ================================================== #

# @undo # Nevermind, the entire bake procedure is contained.
def _match_keys_procedure(pins_to_bake, start_frame, end_frame, composite = True):
    print 'matching between {0} and {1}'.format(start_frame, end_frame)
    for pin in pins_to_bake:
        control = cmds.getAttr(pin + '.control')
        # Snipe translate keys
        translate_keys = cmds.getAttr(pin + '.translate_keys') or []
        if translate_keys:
            translate_keys = [float(x) for x in translate_keys.split(' ')]
            float_translate_keys = translate_keys[:]
            for key in float_translate_keys:
                if not key.is_integer():
                    translate_keys.remove(key)
                    translate_keys.append(int(round(key)))

        translate_keys_baked = set(cmds.keyframe(
                                   control,
                                   attribute = 't',
                                   time = (min(translate_keys or [start_frame]), max(translate_keys or [end_frame])),
                                   query = True) or [])
        translate_keys_to_remove = list(set(translate_keys_baked - \
                                        set(translate_keys)))
        # Snipe rotate keys
        rotate_keys = cmds.getAttr(pin + '.rotate_keys') or []
        if rotate_keys:
            rotate_keys = [float(x) for x in rotate_keys.split(' ')]
            float_rotate_keys = rotate_keys[:]
            for key in float_rotate_keys:
                if not key.is_integer():
                    rotate_keys.remove(key)
                    rotate_keys.append(int(round(key)))

        rotate_keys_baked = set(cmds.keyframe(
                                control,
                                attribute = 'r',
                                time = (min(rotate_keys or [start_frame]), max(rotate_keys or [end_frame])),
                                query = True) or [])
        rotate_keys_to_remove = list(set(rotate_keys_baked - \
                                     set(rotate_keys)))

        keys_baked = list(translate_keys_baked | rotate_keys_baked) # Join 2 sets
        if composite == True:
            composited_keys = list(set(translate_keys + rotate_keys))
            keys_to_remove = list(set(keys_baked) - set(composited_keys))
            # for key in _to_ranges(keys_to_remove): # Had to remove to make room for floats for now
            for key in keys_to_remove:
                cmds.cutKey(control, \
                    t = (key, ), attribute = ('t', 'r'), clear = True)
        else:
            for key in _to_ranges(translate_keys_to_remove):
                cmds.cutKey(control, \
                    t = key, attribute = 't', clear = True)
            for key in _to_ranges(rotate_keys_to_remove):
                cmds.cutKey(control, \
                    t = key, attribute = 'r', clear = True)

        keys_baked.insert(0, keys_baked[0]-1)
        keys_baked.append(keys_baked[-1]+1)
        keys = []
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        for match in tuple_rex.finditer(bp_keys):
            keys.append(float(match.group(0).split(', ')[0]))
            # values = float(match.group(0).split(', ')[1])
        bp_keys_to_remove = list(set(keys_baked) - set(keys))
        for key in _to_ranges(bp_keys_to_remove):
            cmds.cutKey(\
                control, \
                time = key, \
                attribute = ('blendParent1'), \
                clear = True)

    return True

@viewport_off
def _do_bake(nodes_to_bake, start_frame, end_frame, sample = 1):
    '''
    nodes: list
    start_frame: int
    end_frame: int
    '''
    try:
        cmds.bakeResults(
            nodes_to_bake,
            simulation = True,
            time = (start_frame, end_frame),
            sampleBy = sample,
            oversamplingRate = 1,
            disableImplicitControl = True,
            preserveOutsideKeys = True,
            at = ("tx", "ty", "tz", "rx", "ry", "rz", "blendParent1"),
            sparseAnimCurveBake = False,
            removeBakedAttributeFromLayer = False,
            removeBakedAnimFromLayer = False,
            bakeOnOverrideLayer = False,
            minimizeRotation = True,
            controlPoints = False,
            shape = True
            )
        return True
    except:
        return False

# Classes ============================================================ #

class AnimPinUI(QtWidgets.QDialog):
    """docstring for UI"""
    
    UI_INSTANCE = None

    @classmethod
    def run(cls):
        if not cls.UI_INSTANCE:
            cls.UI_INSTANCE = AnimPinUI()

        if cls.UI_INSTANCE.isHidden():
            cls.UI_INSTANCE.show()
        else:
            cls.UI_INSTANCE.raise_()
            cls.UI_INSTANCE.activateWindow()
    
    @classmethod
    def stop(cls):
        if cls.UI_INSTANCE:
            cls.UI_INSTANCE.close()
            # cls.UI_INSTANCE.deleteLater()
            cls.UI_INSTANCE = None

    def __init__(self, parent = None): #_get_maya_window()):
        super(AnimPinUI, self).__init__(parent)
        self.parent = _get_maya_window()
        self.setParent(self.parent)
        self.setWindowFlags(
            QtCore.Qt.Dialog |
            QtCore.Qt.WindowCloseButtonHint #| # Remove the ? button
            # QtCore.Qt.WindowStaysOnTopHint
        )
        self.setObjectName('AnimPin')
        self.setWindowTitle('Animation Pin Tool')
        self.setProperty("saveWindowPref", True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # Class globals
        self.pressPos = None
        self.isMoving = False
        self._callbacks = {}
        self.width  = WIDTH
        self.height = HEIGHT
        self.mini_state = False

        # Organizing the startup sequence
        self.build_UI()
        self.init_connections()
        self.init_frame_range()
        self._init_pin_group_list()



    # Event methods -------------------------------------------------- #

    def init_connections(self):
        # Connections ------------------------------------------------ #
        self.BTN_select_space.clicked.connect(self.on_select_space)
        self.BTN_create_pins.clicked.connect(self.on_create_pins)
        self.BTN_bake_pins.clicked.connect(self.on_bake_pins)
        self.destroyed.connect(self.closeEvent)

        # Listbox connections
        self.LST_pin_groups.itemSelectionChanged.connect(self._pass_selection_to_maya)
        self.LST_pin_groups.itemChanged.connect(self._ui_pin_name_changed)

    def init_frame_range(self):
        spin_start = cmds.playbackOptions(\
            query = True, \
            animationStartTime = True)
        spin_end   = cmds.playbackOptions(\
            query = True, \
            animationEndTime = True)
        self.SPN_start_frame.setValue(spin_start)
        self.SPN_end_frame.setValue(spin_end)

    # Build UI ------------------------------------------------------- #

    def build_UI(self):
        # Start with the stylesheet ---------------------------------- #
        self.setStyleSheet("""
            QWidget{
                background-color: rgb(70, 70, 70);  
                color: rgb(140, 140, 140);
                font: 10pt Arial, Sans-serif;
                outline: 0;
            }
            QGroupBox {
                background-color: rgb(65, 65, 65);
                border: 1px solid;
                border-color: rgb(80, 80, 80); 
                border-radius: 5px;
                margin-top: 2.5ex; 
            }
            QGroupBox::title {
                color:rgb(120,120,120);
                subcontrol-origin: margin;
                subcontrol-position: top center; 
                margin: 0px 4px;
                padding: 0px;
            }
            QLabel#headerLabel{
                background-color: rgb(59, 82, 125);
            }
            Line {
                margin: 0px;
                padding: 0px;
            }
            QSpinBox {
                padding: 0px 8px 0px 5px;
                background-color: rgb(50, 50, 50);
                border-width: 0px;
                border-radius: 8px;
                color: rgb(150, 150, 150);
                font: bold 14pt Sans-serif ;
            }
            QSpinBox:focus {
                background-color: rgb(55, 55, 55);
            }
            QSpinBox:hover {
                background-color: rgb(60, 60, 60);
            }
            QSpinBox:pressed {
                background-color: rgb(74, 105, 129);
            }
            QRadioButton {
                background-color: rgb(65, 65, 65);
                color: rgb(180, 180, 180);
                border-radius:8px;
                padding: 4px;
            }
            QRadioButton:checked{
                background-color: rgb(80, 80, 80); 
            
            }
            QRadioButton:focus {
                background-color: rgb(85, 85, 85);
            }
            QRadioButton:hover{
                background-color: rgb(90, 90, 90);
            }
            QRadioButton:pressed{
                background-color: rgb(74, 105, 129);
            }
            QRadioButton::indicator {
                width:                  8px;
                height:                 8px;
                border-radius:          6px;
            }
            QRadioButton::indicator:checked {
                background-color:   #05B8CC;
                border:             2px solid grey;
                border-color:       rgb(180, 180, 180);
            }
            QRadioButton::indicator:unchecked {
                background-color: rgb(60, 60, 60);
                border:                 2px solid grey;
                border-color: rgb(140, 140, 140);
            }
            QPushButton {
                background-color: rgb(80, 80, 80);
                border-style: solid;
                border-width:0px;
                border-color: rgb(160, 70, 60);
                border-radius:8px;
                color: rgb(186, 186, 186);
                min-height: 50px;
            }
            QPushButton:checked {
                background-color: rgb(157, 102, 71);
            }
            QPushButton:focus {
                background-color: rgb(85, 85, 85);
            }
            QPushButton:hover{
                background-color: rgb(90, 90, 90);
            }
            QPushButton:pressed{
                background-color: rgb(74, 105, 129);
            }
            QPushButton[state='active']{
                background-color: rgb(96, 117, 79);
            }
            QPushButton[state='set']{
                background-color: rgb(70, 99, 91);
            }
            QPushButton[state='clear']{
                background-color: rgb(80, 80, 80);
            }
            QProgressBar {
                border: 1px solid;
                border-color:rgb(90,90,90);
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
            QListWidget {
                show-decoration-selected: 1; 
                background: rgb(65, 65, 65);    
                border: 1px solid grey;
                border-radius: 10px;
                padding: 6px 6px;
                border-color: rgb(80, 80, 80); 
                margin-bottom: 0px;
                padding-right: 6px;
                alternate-background-color: rgb(65, 65, 65); 
            }
            QListWidget:focus {
                background-color: rgb(60, 60, 60);
            }
            QListWidget::item {
                background: rgb(65, 65, 65); 
                margin-bottom: 2px;
                border: 0px solid #000000;
                border-radius: 4px;    
                padding-left: 4px;
                margin-right: 4px;
                height:24px;
            }
            QListWidget::item:alternate {
                border: 0px solid #3c3c3c;
                background: rgb(62, 62, 62);
            }
            QListWidget::item:selected {
                background-color: #4a6981;
            }
            QListWidget::item:selected:!active {
                background-color: #4f718c; 
                color: #fff;
            }
            QListWidget::item:hover {
                background: rgb(80, 80, 80);
            }
            QListWidget::item:selected:hover {
                background-color: #4f7089;
            }
            QListWidget::item::selected:pressed {
                background-color: #648eaf;
            }
            QScrollBar:vertical {
                border: 1px solid grey;
                border-color: rgb(90,90,90);
                background: rgb(60,60,60);
                width: 8px;
                padding:  0;
                margin-right: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgb(90,90,90);
                min-height: 20px;
            }
            QScrollBar::add-line:vertical {
                border: 0px solid grey;
                background: red;
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                border: 0px solid;
                border-color:rgb(90,90,90);
                background: green;
                height: 0px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                border: 1px solid grey;
                width: 10px;
                height: 10px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QAbstractScrollArea::corner {
                background: none;
                border: none;
            }
            QLineEdit {
                color: #fff;
                width: 100%;
                background: rgb(65, 65, 65);    
                border: 1px solid #05B8CC;
                border-radius: 4px;
                padding: 0px 2px;
                height:24px;
            }
            QLineEdit:hover {
                width: 100%;
                background: rgb(65, 65, 65);    
                border: 1px solid #05B8CC;
                border-radius: 4px;
                padding: 0px 2px;
                height:24px;
            }
            QMenu {
                margin: 0px; /* some spacing around the menu */
                background: rgb(65, 65, 65);
                border: 1px solid rgb(115, 115, 115); 
                padding: 8px 8px;
            }
            QMenu::item {
                color: rgb(180, 180, 180);
                padding: 4px 25px 4px 20px;
            }
            QMenu::item:selected {
                background: rgb(45, 45, 45);
                border-radius: 6px;
            }
            QMenu::separator {
                height: 2px;
                margin-left: 10px;
                margin-right: 5px;
            }
            QMenu::indicator {
                width: 13px;
                height: 13px;
            }""")

        # Main layout ------------------------------------------------ #
        self.LYT_main_grid = QtWidgets.QGridLayout()
        self.setLayout(self.LYT_main_grid)
        self.LYT_main_grid.setContentsMargins(0, 0, 0, 0)
        self.LYT_main_grid.setSpacing(6)

        # Header image ----------------------------------------------- #
        qpix = QtGui.QPixmap()
        qpix.loadFromData(image_data)
        self.LBL_header_image = QtWidgets.QLabel()
        self.LYT_main_grid.addWidget(self.LBL_header_image, 0, 0)
        self.LBL_header_image.setObjectName("headerLabel")
        self.LBL_header_image.setPixmap(qpix)
        self.LBL_header_image.setAlignment(QtCore.Qt.AlignCenter)
        self.LBL_header_image.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)

        # The main QVBoxLayout that holds everything under the header -#
        self.LYT_main_vertical = QtWidgets.QVBoxLayout()
        self.LYT_main_grid.addLayout(self.LYT_main_vertical, 1, 0)
        self.LYT_main_vertical.setStretch(1,2)#(1,2,1,1,3,4,0,0)
        self.LYT_main_vertical.setSpacing(8)
        self.LYT_main_vertical.setContentsMargins(10, 4, 10, 10)

        # Construct the start/end frame grid ------------------------- #
        self.LYT_grid_time = QtWidgets.QGridLayout()
        self.LYT_main_vertical.addLayout(self.LYT_grid_time)
        self.LYT_grid_time.setContentsMargins(0, 0, 0, 2)
        self.LYT_grid_time.setColumnStretch(1,2)
        # - LBL_start_frame
        self.LBL_start_frame = QtWidgets.QLabel()
        self.LYT_grid_time.addWidget(self.LBL_start_frame, 0, 0)
        self.LBL_start_frame.setText("Start Frame")
        self.LBL_start_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        self.LBL_start_frame.setMinimumSize(71, 24)
        # - spinStartFrane
        self.SPN_start_frame = QtWidgets.QSpinBox()
        self.LYT_grid_time.addWidget(self.SPN_start_frame, 0, 1)
        self.SPN_start_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        self.SPN_start_frame.setMinimumSize(70, 24)
        self.SPN_start_frame.setAlignment(\
            QtCore.Qt.AlignRight | \
            QtCore.Qt.AlignTrailing | \
            QtCore.Qt.AlignVCenter)
        self.SPN_start_frame.setButtonSymbols(\
            QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_start_frame.setMinimum(-999999999)
        self.SPN_start_frame.setMaximum(999999999)
        # - spinEndFrane
        self.SPN_end_frame = QtWidgets.QSpinBox()
        self.LYT_grid_time.addWidget(self.SPN_end_frame, 1, 1)
        self.SPN_end_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        self.SPN_end_frame.setMinimumSize(70, 24)
        self.SPN_end_frame.setAlignment(\
            QtCore.Qt.AlignRight | \
            QtCore.Qt.AlignTrailing | \
            QtCore.Qt.AlignVCenter)
        self.SPN_end_frame.setButtonSymbols(\
            QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_end_frame.setMinimum(-999999999)
        self.SPN_end_frame.setMaximum(999999999)
        # - LBL_end_frame
        self.LBL_end_frame = QtWidgets.QLabel()
        self.LYT_grid_time.addWidget(self.LBL_end_frame, 1, 0)
        self.LBL_end_frame.setText("End Frame")
        self.LBL_end_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        self.LBL_end_frame.setMinimumSize(71, 24)

        # Line Divider ----------------------------------------------- #
        self.LN_top = QtWidgets.QFrame()
        self.LYT_main_vertical.addWidget(self.LN_top)
        self.LN_top.setFrameShape(QtWidgets.QFrame.HLine)
        self.LN_top.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LN_top.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Fixed)

        # Bake step option box --------------------------------------- #
        self.GRP_options = QtWidgets.QGroupBox()
        self.LYT_main_vertical.addWidget(self.GRP_options)
        self.GRP_options.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        self.GRP_options.setMinimumSize(0, 90)
        self.GRP_options.setTitle("Bake Options")

        self.LYT_options = QtWidgets.QVBoxLayout()
        self.GRP_options.setLayout(self.LYT_options)
        self.LYT_options.setSpacing(6)
        self.LYT_options.setContentsMargins(10, 12, 10, 4)

        self.OPT_0_matchkeys = QtWidgets.QRadioButton()
        self.LYT_options.addWidget(self.OPT_0_matchkeys)
        self.OPT_0_matchkeys.setMinimumSize(0, 24)
        self.OPT_0_matchkeys.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.OPT_0_matchkeys.setText("Match Keys")
        self.OPT_0_matchkeys.setChecked(True)

        self.LYT_step = QtWidgets.QHBoxLayout()
        self.LYT_options.addLayout(self.LYT_step)
        self.LYT_step.setSpacing(6)
        self.LYT_step.setContentsMargins(0, 0, 0, 0)

        self.OPT_1_bakeStep = QtWidgets.QRadioButton()
        self.LYT_step.addWidget(self.OPT_1_bakeStep)
        self.OPT_1_bakeStep.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        self.OPT_1_bakeStep.setMinimumSize(0, 24)
        self.OPT_1_bakeStep.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.OPT_1_bakeStep.setText("Bake step ")

        self.SPN_step = QtWidgets.QSpinBox()
        self.LYT_step.addWidget(self.SPN_step)
        self.SPN_step.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed)
        self.SPN_step.setMinimumSize(20, 24)
        self.SPN_step.setAlignment(\
            QtCore.Qt.AlignRight |\
            QtCore.Qt.AlignTrailing |\
            QtCore.Qt.AlignVCenter)
        self.SPN_step.setButtonSymbols(\
            QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_step.setMinimum(1)
        self.SPN_step.setMaximum(999999999)
        self.SPN_step.setValue(2)

        # Line Divider ----------------------------------------------- #
        self.LN_middle = QtWidgets.QFrame()
        self.LYT_main_vertical.addWidget(self.LN_middle)
        self.LN_middle.setFrameShape(QtWidgets.QFrame.HLine)
        self.LN_middle.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LN_middle.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Fixed)

        # The function buttons --------------------------------------- #
        self.BTN_select_space = QtWidgets.QPushButton()
        self.LYT_main_vertical.addWidget(self.BTN_select_space)
        self.BTN_select_space.setText("Select Space Controls")
        self.BTN_select_space.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred)
        self.BTN_select_space.setMinimumSize(0, 46)
        self.BTN_select_space.setFocusPolicy(QtCore.Qt.WheelFocus)

        self.BTN_create_pins = QtWidgets.QPushButton()
        self.LYT_main_vertical.addWidget(self.BTN_create_pins)
        self.BTN_create_pins.setText("Create Pins")
        self.BTN_create_pins.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred)
        self.BTN_create_pins.setMinimumSize(0, 46)
        self.BTN_create_pins.setFocusPolicy(QtCore.Qt.WheelFocus)

        self.BTN_bake_pins = QtWidgets.QPushButton()
        self.LYT_main_vertical.addWidget(self.BTN_bake_pins)
        self.BTN_bake_pins.setText("Bake Pins")
        self.BTN_bake_pins.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred)
        self.BTN_bake_pins.setMinimumSize(0, 46)
        self.BTN_bake_pins.setEnabled(True)
        self.BTN_bake_pins.setFocusPolicy(QtCore.Qt.WheelFocus)

        # Line Divider ----------------------------------------------- #
        self.LN_bottom = QtWidgets.QFrame()
        self.LYT_main_vertical.addWidget(self.LN_bottom)
        self.LN_bottom.setFrameShape(QtWidgets.QFrame.HLine)
        self.LN_bottom.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LN_bottom.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Fixed)

        # Pin group widget ------------------------------------------- #
        self.LST_pin_groups = QtWidgets.QListWidget()
        self.LYT_main_vertical.addWidget(self.LST_pin_groups)
        self.LST_pin_groups.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        self.LST_pin_groups.setMinimumSize(0, 40)
        self.LST_pin_groups.setVerticalScrollBarPolicy(\
            QtCore.Qt.ScrollBarAsNeeded)
        self.LST_pin_groups.setHorizontalScrollBarPolicy(\
            QtCore.Qt.ScrollBarAlwaysOff)
        self.LST_pin_groups.setEditTriggers(\
            QtWidgets.QAbstractItemView.DoubleClicked |\
            QtWidgets.QAbstractItemView.EditKeyPressed)
        self.LST_pin_groups.setAlternatingRowColors(True)
        self.LST_pin_groups.setSelectionMode(\
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.LST_pin_groups.setWrapping(False)
        self.LST_pin_groups.setResizeMode(\
            QtWidgets.QListView.Adjust)
        self.LST_pin_groups.setLayoutMode(\
            QtWidgets.QListView.SinglePass)
        self.LST_pin_groups.setSpacing(0)
        self.LST_pin_groups.setCurrentRow(-1)

        # Right click menu ------------------------------------------- #

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)

        self.popup_menu = QtWidgets.QMenu(self)
        # self.popup_menu.addAction(QtWidgets.QAction('Options', self))
        # self.popup_menu.addSeparator()
        self.menu_mini = QtWidgets.QAction('Miniaturize UI', self)
        self.popup_menu.addAction(self.menu_mini)
        self.menu_mini.triggered.connect(self.on_menu_mini_clicked)



    # QT Event handling ---------------------------------------------- #

    def mousePressEvent(self, event):
        self.pressPos = event.pos()
        self.isMoving = True
        # if event.button() == QtCore.Qt.RightButton:
            # print "right button clicked"

    def mouseReleaseEvent(self, event):
        self.isMoving = False

    def mouseMoveEvent(self, event):
        if self.isMoving:
            self.newPos = event.pos() - self.pressPos
            self.move(self.window().pos() + self.newPos)

    def showEvent(self, event):
        self.init_callbacks()
        self.resize(self.width, self.height)
        self.update()

    # def hideEvent(self, event):
    #     self.kill_callbacks()

    def closeEvent(self, event):
        self.kill_callbacks()
        global _view
        _view = None
        self.stop()


    # Callback handling ---------------------------------------------- #
    @noUndo
    def init_callbacks(self):
        ''' Get all callbacks installed '''
        self.kill_callbacks(verbose = False)

        # Monitor the changing of all node names. We'll filter out
        # our pin_groups
        nullObj = api.MObject()
        self._callbacks['name_changed'] = \
        api.MNodeMessage.addNameChangedCallback(\
            nullObj, \
            self._maya_node_name_changed)

        # Install master_group handler ------------------------------- #
        global master_group
        if not cmds.objExists(master_group):
            return # Only install if the master_group exists

        master_group_sel = \
        api.MGlobal.getSelectionListByName(master_group).getDagPath(0)

        self._callbacks[master_group + '_dag_added'] = \
        api.MDagMessage.addChildAddedDagPathCallback(\
            master_group_sel, \
            self._pin_group_child_added)

        self._callbacks[master_group + '_dag_removed'] = \
        api.MDagMessage.addChildRemovedDagPathCallback(\
            master_group_sel, \
            self._pin_group_child_removed)

        print "Installed %s callbacks." % master_group


    def kill_callbacks(self, verbose = True):
        if verbose: print "Uninstalling callbacks..."
        for ID in self._callbacks.keys():
            try:
                self._callbacks[ID] = \
                api.MMessage.removeCallback(self._callbacks[ID]) or []
                # if verbose: print "Uninstalled %s callback." % ID
            except:
                if verbose: print "No more callbacks to uninstall."


    # Callback functions --------------------------------------------- #

    def _pin_group_child_added(self, child, parent, client_data):
        cmds.evalDeferred(self._init_pin_group_list)

    def _pin_group_child_removed(self, child, parent, client_data):
        cmds.evalDeferred(self._init_pin_group_list)

    def _maya_node_name_changed(self, mObj, old_name, client_data):
        global master_group
        try:
            item_dag = api.MDagPath.getAPathTo(mObj)
        except:
            return

        if old_name == master_group:
            result = cmds.confirmDialog(\
                backgroundColor = [0.85882, 0.19608, 0.03137],
                icon = "critical",
                title = 'WAIT!',
                message = 'You renamed the animPin master group. '
                          'This breaks everything.',
                messageAlign = "center",
                button = ["I promise I'll undo it!"])

        item_widget = self.LST_pin_groups.findItems(\
            old_name, \
            QtCore.Qt.MatchExactly)
        if not item_widget:
            return
        else:
            wigData = item_widget[0].data(QtCore.Qt.UserRole)
            if wigData == item_dag:
                widget_data_name = wigData.fullPathName().split('|')[-1]
                self.LST_pin_groups.blockSignals(True)
                item_widget[0].setText(widget_data_name)
                self.LST_pin_groups.blockSignals(False)

    def _ui_pin_name_changed(self, item):
        data = item.data(QtCore.Qt.UserRole)
        dataFN = api.MFnDependencyNode(data.node())
        # Prevents this from firing twice if maya doesn't like the name
        self.LST_pin_groups.blockSignals(True)
        dataFN.setName(item.text())
        item.setText(data.fullPathName().split('|')[-1])
        self.LST_pin_groups.blockSignals(False)


    # UI handlers ---------------------------------------------------- #

    @noUndo # Without noundo, it would introduce empty undo calls
    def _init_pin_group_list(self):
        '''Should be run on UI init'''
        # #####
        sel = cmds.ls(selection = True)
        self.LST_pin_groups.clear()
        # cmds.refresh(force=True)
        pin_groups = _get_pin_groups()

        if pin_groups:
            for pin_group in pin_groups:
                # print pin_group
                new_item = QtWidgets.QListWidgetItem(pin_group)
                mSel = api.MGlobal.getSelectionListByName(pin_group)
                item_dag = mSel.getDagPath(0)
                new_item.setData(QtCore.Qt.UserRole, item_dag)
                new_item.setFlags(QtCore.Qt.ItemIsEditable |\
                                 QtCore.Qt.ItemIsSelectable |\
                                 QtCore.Qt.ItemIsEnabled)
                self.LST_pin_groups.addItem(new_item)
                if pin_group in sel:
                    new_item.setSelected(True)

            # self._buffer_list_refresh()
            # self.updateBufferListSelection()


    def _pass_selection_to_maya(self):
        selected_items = []
        selected_items = [x.text() for x in self.LST_pin_groups.selectedItems()]
        #cmds.undoInfo(ock=True) # Turning undo off for the rolling selection
        cmds.select(selected_items, replace=True)
        #cmds.undoInfo(cck=True) # Annoying to have this flood the stack

    # Main button handling ------------------------------------------- #

    def on_select_space(self):
        from klugTools import rig_utils
        rig_utils.select_space_controls()

    def on_create_pins(self):
        self.kill_callbacks(verbose = False)
        new_pin = create_pins(\
            start_frame = self.SPN_start_frame.value(), \
            end_frame = self.SPN_end_frame.value())

        self.init_callbacks()
        self._init_pin_group_list()

    def on_bake_pins(self):
        # Grab SPN_step
        option = 0
        if not self.OPT_0_matchkeys.isChecked():
            option = self.SPN_step.value()
        pin_groups_to_delete = bake_pins(\
            bake_option = option, \
            start_frame = self.SPN_start_frame.value(), \
            end_frame = self.SPN_end_frame.value())
        self._init_pin_group_list()

    # Sub widget handling -------------------------------------------- #

    def on_context_menu(self, point):
        # Show context menu here
        self.popup_menu.exec_(self.mapToGlobal(point))

    def on_menu_mini_clicked(self):
        if self.mini_state == False:
            print "Miniaturizing UI!"
            self.LST_pin_groups.hide()
            self.LN_bottom.hide()
            self.LN_middle.hide()
            self.LN_top.hide()
            self.GRP_options.hide()
            self.LBL_start_frame.hide()
            self.LBL_end_frame.hide()
            self.SPN_start_frame.hide()
            self.SPN_end_frame.hide()
            self.width = self.geometry().width()
            self.height = self.geometry().height()
            self.mini_state = True
            self.menu_mini.setText("Embiggen UI")
        elif self.mini_state == True:
            print "Embiggening UI!"
            self.LST_pin_groups.show()
            self.LN_bottom.show()
            self.LN_middle.show()
            self.LN_top.show()
            self.GRP_options.show()
            self.LBL_start_frame.show()
            self.LBL_end_frame.show()
            self.SPN_start_frame.show()
            self.SPN_end_frame.show()
            self.mini_state = False
            self.menu_mini.setText("Miniaturize UI")
        # self.update()
        QtCore.QTimer.singleShot(0, self.resize_window)

    def resize_window(self):
        # self.resize(180, 180)
        if self.mini_state == False:
            self.resize(self.width, self.height)
        elif self.mini_state == True:
            self.resize(self.minimumSizeHint())


# class Vividict(dict):
#     def __missing__(self, key):
#         value = self[key] = type(self)()
#         return value

# Data =============================================================== #

# Non-beta image:
#image_data = base64.b64decode("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")

# Beta image:
image_data = base64.b64decode("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/GMzLdsAAEKeaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/Pgo8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA1LjYtYzA2NyA3OS4xNTc3NDcsIDIwMTUvMDMvMzAtMjM6NDA6NDIgICAgICAgICI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIKICAgICAgICAgICAgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIgogICAgICAgICAgICB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIKICAgICAgICAgICAgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIKICAgICAgICAgICAgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiCiAgICAgICAgICAgIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPHhtcDpDcmVhdG9yVG9vbD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoV2luZG93cyk8L3htcDpDcmVhdG9yVG9vbD4KICAgICAgICAgPHhtcDpDcmVhdGVEYXRlPjIwMTgtMDMtMTNUMDE6Mzc6MzAtMDc6MDA8L3htcDpDcmVhdGVEYXRlPgogICAgICAgICA8eG1wOk1ldGFkYXRhRGF0ZT4yMDE4LTA4LTMxVDE3OjA4OjQyLTA3OjAwPC94bXA6TWV0YWRhdGFEYXRlPgogICAgICAgICA8eG1wOk1vZGlmeURhdGU+MjAxOC0wOC0zMVQxNzowODo0Mi0wNzowMDwveG1wOk1vZGlmeURhdGU+CiAgICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2UvcG5nPC9kYzpmb3JtYXQ+CiAgICAgICAgIDx4bXBNTTpJbnN0YW5jZUlEPnhtcC5paWQ6ZjZiMWUxZWEtZDRlNC01NTRkLTkzZDYtYzUzNzlhODhiM2VhPC94bXBNTTpJbnN0YW5jZUlEPgogICAgICAgICA8eG1wTU06RG9jdW1lbnRJRD5hZG9iZTpkb2NpZDpwaG90b3Nob3A6MjM3MjgwN2MtYWQ3Yi0xMWU4LTgzNzQtZGI1M2Q5ZWQwNzNlPC94bXBNTTpEb2N1bWVudElEPgogICAgICAgICA8eG1wTU06T3JpZ2luYWxEb2N1bWVudElEPnhtcC5kaWQ6Nzk4MTg5ZDUtMDFhYi04NDRlLTlhMDUtOWU0ZmFiMDNiYzFmPC94bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ+CiAgICAgICAgIDx4bXBNTTpIaXN0b3J5PgogICAgICAgICAgICA8cmRmOlNlcT4KICAgICAgICAgICAgICAgPHJkZjpsaSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDphY3Rpb24+Y3JlYXRlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6aW5zdGFuY2VJRD54bXAuaWlkOjc5ODE4OWQ1LTAxYWItODQ0ZS05YTA1LTllNGZhYjAzYmMxZjwvc3RFdnQ6aW5zdGFuY2VJRD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OndoZW4+MjAxOC0wMy0xM1QwMTozNzozMC0wNzowMDwvc3RFdnQ6d2hlbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnNvZnR3YXJlQWdlbnQ+QWRvYmUgUGhvdG9zaG9wIENDIDIwMTUgKFdpbmRvd3MpPC9zdEV2dDpzb2Z0d2FyZUFnZW50PgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgICAgPHJkZjpsaSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDphY3Rpb24+c2F2ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0Omluc3RhbmNlSUQ+eG1wLmlpZDpiMWFhNjVjNC1mZGRkLThhNDMtOGFlNC0yMzViOGY4NmQwYzg8L3N0RXZ0Omluc3RhbmNlSUQ+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDp3aGVuPjIwMTgtMDMtMTNUMDE6Mzk6NTUtMDc6MDA8L3N0RXZ0OndoZW4+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDpzb2Z0d2FyZUFnZW50PkFkb2JlIFBob3Rvc2hvcCBDQyAyMDE1IChXaW5kb3dzKTwvc3RFdnQ6c29mdHdhcmVBZ2VudD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmNoYW5nZWQ+Lzwvc3RFdnQ6Y2hhbmdlZD4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPnNhdmVkPC9zdEV2dDphY3Rpb24+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDppbnN0YW5jZUlEPnhtcC5paWQ6MWM2MjhjM2UtZjYxZi1lYTRmLWI0NzgtNzViNGU1ODM4Nzc4PC9zdEV2dDppbnN0YW5jZUlEPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6d2hlbj4yMDE4LTA4LTMxVDE3OjA4OjQyLTA3OjAwPC9zdEV2dDp3aGVuPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6c29mdHdhcmVBZ2VudD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoV2luZG93cyk8L3N0RXZ0OnNvZnR3YXJlQWdlbnQ+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDpjaGFuZ2VkPi88L3N0RXZ0OmNoYW5nZWQ+CiAgICAgICAgICAgICAgIDwvcmRmOmxpPgogICAgICAgICAgICAgICA8cmRmOmxpIHJkZjpwYXJzZVR5cGU9IlJlc291cmNlIj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmFjdGlvbj5jb252ZXJ0ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnBhcmFtZXRlcnM+ZnJvbSBhcHBsaWNhdGlvbi92bmQuYWRvYmUucGhvdG9zaG9wIHRvIGltYWdlL3BuZzwvc3RFdnQ6cGFyYW1ldGVycz4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPmRlcml2ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnBhcmFtZXRlcnM+Y29udmVydGVkIGZyb20gYXBwbGljYXRpb24vdm5kLmFkb2JlLnBob3Rvc2hvcCB0byBpbWFnZS9wbmc8L3N0RXZ0OnBhcmFtZXRlcnM+CiAgICAgICAgICAgICAgIDwvcmRmOmxpPgogICAgICAgICAgICAgICA8cmRmOmxpIHJkZjpwYXJzZVR5cGU9IlJlc291cmNlIj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmFjdGlvbj5zYXZlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6aW5zdGFuY2VJRD54bXAuaWlkOmY2YjFlMWVhLWQ0ZTQtNTU0ZC05M2Q2LWM1Mzc5YTg4YjNlYTwvc3RFdnQ6aW5zdGFuY2VJRD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OndoZW4+MjAxOC0wOC0zMVQxNzowODo0Mi0wNzowMDwvc3RFdnQ6d2hlbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnNvZnR3YXJlQWdlbnQ+QWRvYmUgUGhvdG9zaG9wIENDIDIwMTUgKFdpbmRvd3MpPC9zdEV2dDpzb2Z0d2FyZUFnZW50PgogICAgICAgICAgICAgICAgICA8c3RFdnQ6Y2hhbmdlZD4vPC9zdEV2dDpjaGFuZ2VkPgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgPC9yZGY6U2VxPgogICAgICAgICA8L3htcE1NOkhpc3Rvcnk+CiAgICAgICAgIDx4bXBNTTpEZXJpdmVkRnJvbSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgIDxzdFJlZjppbnN0YW5jZUlEPnhtcC5paWQ6MWM2MjhjM2UtZjYxZi1lYTRmLWI0NzgtNzViNGU1ODM4Nzc4PC9zdFJlZjppbnN0YW5jZUlEPgogICAgICAgICAgICA8c3RSZWY6ZG9jdW1lbnRJRD5hZG9iZTpkb2NpZDpwaG90b3Nob3A6ZDAxMjIxZjktM2JjNy0xMWU4LThjZmItZTllNzY2YThmODk1PC9zdFJlZjpkb2N1bWVudElEPgogICAgICAgICAgICA8c3RSZWY6b3JpZ2luYWxEb2N1bWVudElEPnhtcC5kaWQ6Nzk4MTg5ZDUtMDFhYi04NDRlLTlhMDUtOWU0ZmFiMDNiYzFmPC9zdFJlZjpvcmlnaW5hbERvY3VtZW50SUQ+CiAgICAgICAgIDwveG1wTU06RGVyaXZlZEZyb20+CiAgICAgICAgIDxwaG90b3Nob3A6Q29sb3JNb2RlPjM8L3Bob3Rvc2hvcDpDb2xvck1vZGU+CiAgICAgICAgIDxwaG90b3Nob3A6SUNDUHJvZmlsZT5zUkdCIElFQzYxOTY2LTIuMTwvcGhvdG9zaG9wOklDQ1Byb2ZpbGU+CiAgICAgICAgIDxwaG90b3Nob3A6VGV4dExheWVycz4KICAgICAgICAgICAgPHJkZjpCYWc+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8cGhvdG9zaG9wOkxheWVyTmFtZT5hbmltUGluPC9waG90b3Nob3A6TGF5ZXJOYW1lPgogICAgICAgICAgICAgICAgICA8cGhvdG9zaG9wOkxheWVyVGV4dD5hbmltUGluPC9waG90b3Nob3A6TGF5ZXJUZXh0PgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgPC9yZGY6QmFnPgogICAgICAgICA8L3Bob3Rvc2hvcDpUZXh0TGF5ZXJzPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICAgICA8dGlmZjpYUmVzb2x1dGlvbj4zMDAwMDAwLzEwMDAwPC90aWZmOlhSZXNvbHV0aW9uPgogICAgICAgICA8dGlmZjpZUmVzb2x1dGlvbj4zMDAwMDAwLzEwMDAwPC90aWZmOllSZXNvbHV0aW9uPgogICAgICAgICA8dGlmZjpSZXNvbHV0aW9uVW5pdD4yPC90aWZmOlJlc29sdXRpb25Vbml0PgogICAgICAgICA8ZXhpZjpDb2xvclNwYWNlPjE8L2V4aWY6Q29sb3JTcGFjZT4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjE4MDwvZXhpZjpQaXhlbFhEaW1lbnNpb24+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj41MDwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo8P3hwYWNrZXQgZW5kPSJ3Ij8+dyLjjQAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAA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")

# Public call ======================================================== #

def UI():
    AnimPinUI.run()
    
# Developer section ================================================== #

if __name__ == '__main__':
    AnimPinUI.stop()
    
    UI()

    
# Eof
