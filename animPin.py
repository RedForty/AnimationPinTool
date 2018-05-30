# Animation Pin Tool
# -------------------------------------------------------------------- #

__author__  = "Daniel Klug"
__version__ = "1.10"
__date__    = "04-26-2018"
__email__   = "daniel@redforty.com"

# -------------------------------------------------------------------- #
'''
Installation:
Place both files in your maya/scripts folder, then restart Maya

Usage:
To launch the UI, use this python command:
import animPin as animPin; animPin.show()

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

# Decorators ========================================================= #

def viewportOff(func):
    """
    Decorator - turn off Maya display while func is running.
    if func will fail, the error will be raised after.
    """
    @wraps(func)
    def wrap( *args, **kwargs ):

        # Turn $gMainPane Off:
        mel.eval("paneLayout -e -manage false $gMainPane")
        cmds.refresh(suspend=True)

        # Decorator will try/except running the function.
        # But it will always turn on the viewport at the end.
        # In case the function failed, it will prevent leaving maya viewport off.
        try:
            return func( *args, **kwargs )
        except Exception:
            raise # will raise original error
        finally:
            mel.eval("paneLayout -e -manage true $gMainPane")
            cmds.refresh(suspend=False)
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

        if 'animation' in controlFN.classification(controlFN.typeName):
            continue # Quietly ignore animation curves

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
    for pin in pins_to_bake:
        control = cmds.getAttr(pin + '.control')
        # Snipe translate keys
        translate_keys = cmds.getAttr(pin + '.translate_keys') or []
        translate_keys_baked = set(cmds.keyframe(
                                   control,
                                   attribute = 't',
                                   time = (start_frame, end_frame),
                                   query = True) or [])
        if translate_keys:
            translate_keys = [float(x) for x in translate_keys.split(' ')]
            float_translate_keys = translate_keys[:]
            for key in float_translate_keys:
                if not key.is_integer():
                    translate_keys.remove(key)
                    translate_keys.append(int(round(key)))
        translate_keys_to_remove = list(set(translate_keys_baked - \
                                        set(translate_keys)))
        # Snipe rotate keys
        rotate_keys = cmds.getAttr(pin + '.rotate_keys') or []
        rotate_keys_baked = set(cmds.keyframe(
                                control,
                                attribute = 'r',
                                time = (start_frame, end_frame),
                                query = True) or [])
        if rotate_keys:
            rotate_keys = [float(x) for x in rotate_keys.split(' ')]
            float_rotate_keys = rotate_keys[:]
            for key in float_rotate_keys:
                if not key.is_integer():
                    rotate_keys.remove(key)
                    rotate_keys.append(int(round(key)))                    
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

@viewportOff
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

class View(QtWidgets.QDialog):
    """docstring for View"""

    def __init__(self, parent = None): #_get_maya_window()):
        super(View, self).__init__(parent)
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
        self.setStyleSheet("\
            QWidget{\
                background-color: rgb(70, 70, 70);  \
                color: rgb(140, 140, 140);\
                font: 10pt Arial, Sans-serif;\
                outline: 0;\
            }\
            QGroupBox {\
                background-color: rgb(65, 65, 65);\
                border: 1px solid;\
                border-color: rgb(80, 80, 80); \
                border-radius: 5px;\
                margin-top: 2.5ex; \
            }\
            QGroupBox::title {\
                color:rgb(120,120,120);\
                subcontrol-origin: margin;\
                subcontrol-position: top center; \
                margin: 0px 4px;\
                padding: 0px;\
            }\
            QLabel#headerLabel{\
                background-color: rgb(59, 82, 125);\
            }\
            Line {\
                margin: 0px;\
                padding: 0px;\
            }\
            QSpinBox {\
                padding: 0px 8px 0px 5px;\
                background-color: rgb(50, 50, 50);\
                border-width: 0px;\
                border-radius: 8px;\
                color: rgb(150, 150, 150);\
                font: bold 14pt Sans-serif ;\
            }\
            QSpinBox:focus {\
                background-color: rgb(55, 55, 55);\
            }\
            QSpinBox:hover {\
                background-color: rgb(60, 60, 60);\
            }\
            QSpinBox:pressed {\
                background-color: rgb(74, 105, 129);\
            }\
            QRadioButton {\
                background-color: rgb(65, 65, 65);\
                color: rgb(180, 180, 180);\
                border-radius:8px;\
                padding: 4px;\
            }\
            QRadioButton:checked{\
                background-color: rgb(80, 80, 80); \
            \
            }\
            QRadioButton:focus {\
                background-color: rgb(85, 85, 85);\
            }\
            QRadioButton:hover{\
                background-color: rgb(90, 90, 90);\
            }\
            QRadioButton:pressed{\
                background-color: rgb(74, 105, 129);\
            }\
            QRadioButton::indicator {\
                width:                  8px;\
                height:                 8px;\
                border-radius:          6px;\
            }\
            QRadioButton::indicator:checked {\
                background-color:   #05B8CC;\
                border:             2px solid grey;\
                border-color:       rgb(180, 180, 180);\
            }\
            QRadioButton::indicator:unchecked {\
                background-color: rgb(60, 60, 60);\
                border:                 2px solid grey;\
                border-color: rgb(140, 140, 140);\
            }\
            QPushButton {\
                background-color: rgb(80, 80, 80);\
                border-style: solid;\
                border-width:0px;\
                border-color: rgb(160, 70, 60);\
                border-radius:8px;\
                color: rgb(186, 186, 186);\
                min-height: 50px;\
            }\
            QPushButton:checked {\
                background-color: rgb(157, 102, 71);\
            }\
            QPushButton:focus {\
                background-color: rgb(85, 85, 85);\
            }\
            QPushButton:hover{\
                background-color: rgb(90, 90, 90);\
            }\
            QPushButton:pressed{\
                background-color: rgb(74, 105, 129);\
            }\
            QPushButton[state='active']{\
                background-color: rgb(96, 117, 79);\
            }\
            QPushButton[state='set']{\
                background-color: rgb(70, 99, 91);\
            }\
            QPushButton[state='clear']{\
                background-color: rgb(80, 80, 80);\
            }\
            QProgressBar {\
                border: 1px solid;\
                border-color:rgb(90,90,90);\
                border-radius: 5px;\
            }\
            QProgressBar::chunk {\
                background-color: #05B8CC;\
                width: 20px;\
            }\
            QListWidget {\
                show-decoration-selected: 1; \
                background: rgb(65, 65, 65);    \
                border: 1px solid grey;\
                border-radius: 10px;\
                padding: 6px 6px;\
                border-color: rgb(80, 80, 80); \
                margin-bottom: 0px;\
                padding-right: 6px;\
                alternate-background-color: rgb(65, 65, 65); \
            }\
            QListWidget:focus {\
                background-color: rgb(60, 60, 60);\
            }\
            QListWidget::item {\
                background: rgb(65, 65, 65); \
                margin-bottom: 2px;\
                border: 0px solid #000000;\
                border-radius: 4px;    \
                padding-left: 4px;\
                margin-right: 4px;\
                height:24px;\
            }\
            QListWidget::item:alternate {\
                border: 0px solid #3c3c3c;\
                background: rgb(62, 62, 62);\
            }\
            QListWidget::item:selected {\
                background-color: #4a6981;\
            }\
            QListWidget::item:selected:!active {\
                background-color: #4f718c; \
                color: #fff;\
            }\
            QListWidget::item:hover {\
                background: rgb(80, 80, 80);\
            }\
            QListWidget::item:selected:hover {\
                background-color: #4f7089;\
            }\
            QListWidget::item::selected:pressed {\
                background-color: #648eaf;\
            }\
            QScrollBar:vertical {\
                border: 1px solid grey;\
                border-color: rgb(90,90,90);\
                background: rgb(60,60,60);\
                width: 8px;\
                padding:  0;\
                margin-right: 0px;\
            }\
            QScrollBar::handle:vertical {\
                background: rgb(90,90,90);\
                min-height: 20px;\
            }\
            QScrollBar::add-line:vertical {\
                border: 0px solid grey;\
                background: red;\
                height: 0px;\
                subcontrol-position: bottom;\
                subcontrol-origin: margin;\
            }\
            QScrollBar::sub-line:vertical {\
                border: 0px solid;\
                border-color:rgb(90,90,90);\
                background: green;\
                height: 0px;\
                subcontrol-position: top;\
                subcontrol-origin: margin;\
            }\
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\
                border: 1px solid grey;\
                width: 10px;\
                height: 10px;\
            }\
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\
                background: none;\
            }\
            QAbstractScrollArea::corner {\
                background: none;\
                border: none;\
            }\
            QLineEdit {\
                color: #fff;\
                width: 100%;\
                background: rgb(65, 65, 65);    \
                border: 1px solid #05B8CC;\
                border-radius: 4px;\
                padding: 0px 2px;\
                height:24px;\
            }\
            QLineEdit:hover {\
                width: 100%;\
                background: rgb(65, 65, 65);    \
                border: 1px solid #05B8CC;\
                border-radius: 4px;\
                padding: 0px 2px;\
                height:24px;\
            }\
            QMenu {\
                margin: 0px; /* some spacing around the menu */\
                background: rgb(65, 65, 65);\
                border: 1px solid rgb(115, 115, 115); \
                padding: 8px 8px;\
            }\
            QMenu::item {\
                color: rgb(180, 180, 180);\
                padding: 4px 25px 4px 20px;\
            }\
            QMenu::item:selected {\
                background: rgb(45, 45, 45);\
                border-radius: 6px;\
            }\
            QMenu::separator {\
                height: 2px;\
                margin-left: 10px;\
                margin-right: 5px;\
            }\
            QMenu::indicator {\
                width: 13px;\
                height: 13px;\
            }\
            ")

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

image_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAALQAAAAyCAYAAAD1JPH3AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAC4jAAAuIwF4pT92AAAFRklEQVR42u3db0wbZRwH8O+MCYPGUJGysZGuBGtBkg4kEbBaVLTIgpsRQgJZEAJZNHFTjBqHAaMu6iaasCXGFy7osoyEFBc02ezi4iBhDhWW1QzKGv5dCArFrYS0wCt9Ydr16ZUbNJR7ePh93t3Th7vnnvvm7nmeu4Rtlpeb/wUhgrhP7QYQsp4o0EQoFGgiFAo0EQoFmgiFAk2EQoEmQqFAE6FQoIlQKNBEKBRoIpT71W4Az2yWLGb7+k0JHq9P7WZt2fNYDQq0gpa3qpjtj77swKW+4Q05ts2SJTu+EpdbQt/vw2jv6uPqPDYaDTkEkWnUo766BOfaXofJoFO7OaqhQAtGn5aC4+/XQKfVqN0UVdCQQ4HLLTHb8wt+VdsjTc3Cv7gkK8806pnt5KRENDaUoqnVzuV5xBIFWkHDe9+o3QTGt52XI459dVoNGhtKYS00B8ushWbotBfh8fq4O49Y2hSBzjfvQX7Ow8gwpEKTsD1Y7hwah2t0WnaRdVoNcrPv3rXmF/zod04CAOrKLTCmpyIlWQsAmJ3zYvDPMdgdg7LjKq0OhP8WaEP4/scm/0bXxX6MTHiCdcPrOIfG4eh1MnXWwuP1oanVjnNtO6FPSwmWF+WbYHcMrnge69VPPOE60CaDDu+8ekD2SA0IlNdWzqLt9I/Bi5GbrWdm9i63hDGpA20f1jEXPLAPa6EZLzzzGI5+1sEsZymtDoT/dv3m8RX3/7RlL5pPnEW/cxKfvF3B3EkDdcpsBeg4fyXiKsVqXRsYZo6/a8eDiuexXv3EE24nhTqtBqeOHVoxzKH0aSk4erhCcSIU6SKFyjTq0dhQGnV7GxtKV9x/QnwcPn73IJpeK5OFObROfXWJ7G660WLdT7HGbaCr9hcgIT6OKXO5JXR296Czu0c20UlOSkTV/oKI+8o06hUvUoC10Bz1ktdKQQ1IiI/DvuceBwD4F5fhckvwLy7L6tVWFkfdZ88+lcNsL/iW1vT3G9FPscbtkKMgj71T9f7qDM7aA47UFKPyQFFw2/xouuI+O7t70PHDNXi8PpgMOrxSUSQLYonVjJGJy2tur39xmRky1JVbUF9dIqs3cOMWjp08HxzDht8R9Wkpaw5LYFKYnJTIlF8duBVV38eyn2KN20CH3ym+s/fI6rhGp5ltpeFJZ3cPTp65ewFGJjxoarXj0lmT7EkQjSt9N5jxb3tXH3Ky05G39xGm3ldnHMHxp8frw/cXruLNQy8xdfbsTo54jNrKYlS+aJGVRzpvl1uKapIZ636KNW4D/WR5i+LvNkvWmh7Pjl5nxHJpamZV4/R7+cM5KisbnfiLCXSkkNkdg7JAr2Q1wwHg/6fF5193R3Uese6nWOM20KFslixkZuxChiEVuoe0q76woaJdEtts5m7P49NT9qjPd7P3E9eBPlJTjDJbwaZ41KlN6eOkrYTbQIdP+EK53BKcQ+NY8C1FnHiJSuSv5NYLl4E2GXSyMLvcEn76ZZB5U6X2mi3hD5eBfiKPXRmYuz0f8e3U7p1JajeVcIbLFysPaLYz23P/zEd81fq8NVftphLOcBnocPq0HczLBpNBh7aWg1GtdhCxcTnkmJ65w2wnxMfh9BeHg6+7N8N6KFEHl3dou2MQ0tSsrDzTqGfCHOlbCLK1cRloAHjjg3bZB0ihLvz8G5pPnFW7mYQz23j/lxSBj/sDpmfuoKd/hNvvcYm6uBxDh+p3TgY/3CfkXrgdchASDQo0EQoFmgiFAk2EQoEmQqFAE6FQoIlQKNBEKBRoIhQKNBEKBZoIhQJNhEKBJkL5D+BBAKq9JfatAAAAAElFTkSuQmCC")

# Footer ============================================================= #

def show():
    global _view
    if _view is None:
        _view = View()

    _view.show()

# Eof
