# Animation Pin Tool
# -------------------------------------------------------------------- #

__author__  = "Daniel Klug"
__version__ = "1.03"
__date__    = "04-24-2018"
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

try:
    import sip # Try PyQt4 or 5
except ImportError:
    import shiboken  # Do Pyside or Pyside2

# Qt is a project by Marcus Ottosson-> https://github.com/mottosso/Qt.py
from Qt import QtGui, QtCore, QtCompat, QtWidgets, __binding__
# from Qt.QtGui import QPen, QColor, QBrush, QLinearGradient

import base64
import itertools
from collections import OrderedDict
from functools import wraps
import re # I'm so sorry

# Globals ============================================================ #

_view = None
ap_prefix= '_pin'
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
    master_group = _refresh_group_setup(group_override) # Set the stage

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

    new_pin_group = _create_new_pin_group(master_group)
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


    print "Success!"

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
        controls_to_bake.append(control)
        pin_parent = cmds.listRelatives(pin, parent=True)[0]
        pin_groups_to_delete.add(pin_parent)
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        blendParents_to_restore[control] = bp_keys

    start_frame, end_frame = _validate_bakerange(pins_to_bake, start_frame, end_frame)
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
    success = _do_bake(controls_to_bake, start_frame, end_frame, sample)
    if not success:
        raise ValueError("Bake failed.")
    cmds.refresh() # Check it out

    for control, bp_keys in blendParents_to_restore.items():
        for match in tuple_rex.finditer(bp_keys):
            cmds.setKeyframe(\
                control, \
                at='blendParent1', \
                time = float(match.group(0).split(', ')[0]), \
                value = float(match.group(0).split(', ')[1]) \
            )

    if bake_option == 0: # Proceed with the Match Keys procedure
        success = _match_keys_procedure(pins_to_bake, start_frame, end_frame)
        if not success:
            raise ValueError("match_keys_procedure failed.") 

    cmds.delete(constraints)    
    cmds.delete(list(pin_groups_to_delete))
    print "Success!"
    return controls_to_bake # We're done here!


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
    selected_range = [int(x) for x in selected_range.strip('"').split(':')]
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
    if any(plug.child(p).isLocked for p in range(plug.numChildren())):
        return True
    if not any(plug.child(p).isKeyable for p in range(plug.numChildren())):
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
        control_dep = sel_list.getDependNode(i)
        controlFN = api.MFnDependencyNode(control_dep)
        control_name = controlFN.name()

        pinned_control = control_name + ap_prefix
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

        if not controlFN.typeName in ['transform', 'joint']:
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
    found_pin_groups = []
    master_group = _refresh_group_setup()
    master_group_children = cmds.listRelatives(master_group, type = 'transform') or []
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
    if not pin_groups:
        return []
    for group in pin_groups:
        pins = cmds.listRelatives(group, type = 'transform') or []
        for pin in pins:
            found_pins.append(pin)
    return found_pins


def _validate_framerange(start_frame, end_frame):
    if start_frame == None:
        api.MGlobal.displayWarning(\
            "No start_frame supplied. Defaulting to timeline...")
        start_frame = cmds.playbackOptions(q=True, minTime=True)

    if end_frame == None:
        api.MGlobal.displayWarning(\
            "No end_frame supplied. Defaulting to timeline...")
        end_frame = cmds.playbackOptions(q=True, maxTime=True)

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
        bp_key_times = _get_keys_from_obj_attribute(controlFN, 'blendParent1')
        if bp_key_times:
            key_values = []
            for time in bp_key_times:
                key_value = cmds.getAttr(control_name + '.blendParent1', time = time)
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


def _refresh_group_setup(group_override = None):
    global master_group
    if isinstance(group_override, str):
        master_group = group_override
    if not cmds.objExists(master_group):
        master_group = cmds.createNode('transform', 
                                   name = master_group, 
                                   skipSelect = True)
    return master_group


def _create_new_pin_group(master_group):
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

    locator = cmds.spaceLocator(name = control_name + ap_prefix)[0]
    cmds.setAttr(locator + ".scale", *[locator_scale]*3) # Unpack * 3
    cmds.parent(locator, pin_group)

    for attr in ['x', 'y', 'z']:
        cmds.setAttr(locator + '.s' + attr, keyable=False, channelBox=True)

    for key, value in pin_data.iteritems():
        cmds.addAttr(locator, longName=key, **value) # kwargs ftw

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

# @undo
def _match_keys_procedure(pins_to_bake, start_frame, end_frame, composite = True):
    for pin in pins_to_bake:
        control = cmds.getAttr(pin + '.control')
        # Snipe translate keys
        translate_keys = cmds.getAttr(pin + '.translate_keys') or []
        if translate_keys:
            translate_keys = [float(x) for x in translate_keys.split(' ')]
        translate_keys_baked = set(cmds.keyframe(
                                   control, 
                                   attribute='t', 
                                   time=(start_frame, end_frame), 
                                   query=True) or [])
        translate_keys_to_remove = list(set(translate_keys_baked - \
                                        set(translate_keys)))

        # Snipe rotate keys
        rotate_keys = cmds.getAttr(pin + '.rotate_keys') or []
        if rotate_keys:
            rotate_keys = [float(x) for x in rotate_keys.split(' ')]
        rotate_keys_baked = set(cmds.keyframe(
                                control, 
                                attribute='r', 
                                time=(start_frame, end_frame), 
                                query=True) or [])
        rotate_keys_to_remove = list(set(rotate_keys_baked - \
                                     set(rotate_keys)))

        keys_baked = list(translate_keys_baked | rotate_keys_baked) # Join 2 sets
        if composite == True:
            composited_keys = list(set(translate_keys + rotate_keys))
            keys_to_remove = list(set(keys_baked) - set(composited_keys))
            for key in _to_ranges(keys_to_remove):
                cmds.cutKey(control, t=key, attribute=('t', 'r'), clear=True)
            # for key in _to_ranges(keys_to_remove):
            #     cmds.cutKey(control, t=key, attribute='r', clear=True)    
        else:
            for key in _to_ranges(translate_keys_to_remove):
                cmds.cutKey(control, t=key, attribute='t', clear=True)
            for key in _to_ranges(rotate_keys_to_remove):
                cmds.cutKey(control, t=key, attribute='r', clear=True)

        keys_baked.insert(0, keys_baked[0]-1)
        keys_baked.append(keys_baked[-1]+1)
        keys = []
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        for match in tuple_rex.finditer(bp_keys):
            keys.append(float(match.group(0).split(', ')[0]))
            # values = float(match.group(0).split(', ')[1])
        bp_keys_to_remove = list(set(keys_baked) - set(keys))
        for key in _to_ranges(bp_keys_to_remove):
            cmds.cutKey(control, t=key, attribute=('blendParent1'), clear=True)



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
                         disableImplicitControl = False,
                         preserveOutsideKeys = True,
                         minimizeRotation = True,
                         at = ("tx", "ty", "tz", "rx", "ry", "rz")
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
            QtCore.Qt.WindowSystemMenuHint #| # Remove the ? button
            # QtCore.Qt.WindowStaysOnTopHint
        )
        self.setObjectName('AnimPin')
        self.setWindowTitle('Animation Pin Tool')
        self.setProperty("saveWindowPref", True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # Build UI --------------------------------------------------- #
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(20, 20, 20, 20)
        self.layout().setSpacing(8)
        self.setStyleSheet("\
            QWidget {\
                background-color: rgb(70, 70, 70);\
                color: rgb(140, 140, 140);\
                font: 10pt Helvetica, sans-serif;\
                outline: 0;\
            }\
            QLabel {\
                background-color: rgb(59, 82, 125);\
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
            ")

        qpix = QtGui.QPixmap()
        qpix.loadFromData(image_data)

        label = QtWidgets.QLabel()
        label.setPixmap(qpix)
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout().addWidget(label)

        BTN_create_pins = QtWidgets.QPushButton()
        BTN_create_pins.setText("Create Pins")
        BTN_create_pins.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.layout().addWidget(BTN_create_pins)

        BTN_bake_pins = QtWidgets.QPushButton()
        BTN_bake_pins.setText("Bake Pins")
        BTN_bake_pins.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.layout().addWidget(BTN_bake_pins)

        # Connections ------------------------------------------------ #
        BTN_create_pins.clicked.connect(self.on_create_pins)
        BTN_bake_pins.clicked.connect(self.on_bake_pins)
        self.destroyed.connect(self.closeEvent)

    # Event methods -------------------------------------------------- #
        self.pressPos = None
        self.isMoving = False

    def mousePressEvent(self, event):
        self.pressPos = event.pos()
        self.isMoving = True

    def mouseMoveEvent(self, event):
        if self.isMoving:
            self.newPos = event.pos() - self.pressPos
            self.move(self.window().pos() + self.newPos)

    # QT Event handling ---------------------------------------------- #

    def showEvent(self, event):
        self.init_callbacks()

    # def hideEvent(self, event):
    #     self.kill_callbacks()

    def closeEvent(self, event):
        self.kill_callbacks()
        global _view
        _view = None

    # Callback handling ---------------------------------------------- #

    def init_callbacks(self):
        print "Initializing callbacks"

    def kill_callbacks(self):
        print "killing all callbacks"

    # Widget handling ------------------------------------------------ #
    
    def on_create_pins(self):
        create_pins()

    def on_bake_pins(self):
        bake_pins(bake_option = 0)

    def print_text(self, payload):
        print payload

# Data =============================================================== #

image_data = base64.b64decode("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")

# Footer ============================================================= #

def show():
    global _view
    if _view is None:
        _view = View()

    _view.show()

# Eof
