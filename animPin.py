# Animation Pin Tool - Single-file Distribution
# -------------------------------------------------------------------- #

"""
Animation Pin Tool - Combined Distribution

This is a single-file distribution that combines the core animation
library and the Qt-based UI into one file for easy deployment.

Usage:
    import animPin
    animPin.show()           # Launch the UI
    animPin.create_pins()    # Create pins via command line
    animPin.bake_pins()      # Bake pins via command line
"""

__author__ = "Daniel Klug"
__version__ = "2.0.3"
__date__ = "2026-01-01"
__email__ = "daniel@redforty.com"

# =============================================================================
# Imports
# =============================================================================

import base64
import math
import time
import itertools
import re
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, List, Dict, Callable

import maya.cmds as cmds
import maya.mel as mel
import maya.api.OpenMaya as api
import maya.api.OpenMayaAnim as anim
import maya.OpenMayaUI as mui
import numpy as np

# Qt is a project by Marcus Ottosson -> https://github.com/mottosso/Qt.py
from Qt import QtGui, QtCore, QtCompat, QtWidgets

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# =============================================================================
# Constants and Globals
# =============================================================================

# Pin tool constants
QUICKSELECTSET = 'animPins'
AP_SUFFIX = '_pin'
PIN_GROUP = 'pin_group#'
MASTER_GROUP = 'animPins_group'
LOCATOR_SCALE = 10  # Season to taste - depends on your rig size
LOCATOR_COLOR = (0.273, 0.432, 0.152)  # Light green

# Pin data structure template
PIN_DATA = OrderedDict([
    ('control', {'dataType': "string"}),
    ('constraint', {'dataType': "string"}),
    ('start_frame', {'attributeType': "float"}),
    ('end_frame', {'attributeType': "float"}),
    ('translate_keys', {'dataType': "string"}),
    ('rotate_keys', {'dataType': "string"}),
    ('translate_locked', {'attributeType': "bool"}),
    ('rotate_locked', {'attributeType': "bool"}),
    ('preserve_blendParent', {'dataType': "string"})
])

# Get the timeline object
aPlayBackSliderPython = mel.eval('$tmpVar=$gPlayBackSlider')
tuple_rex = re.compile(r"([0-9]+\.[0-9]+\, [0-9]+\.[0-9]+)")

# Attribute mappings for layer conversion
SHORT_TO_LONG = {
    'tx': 'translateX', 'ty': 'translateY', 'tz': 'translateZ',
    'rx': 'rotateX', 'ry': 'rotateY', 'rz': 'rotateZ',
    'sx': 'scaleX', 'sy': 'scaleY', 'sz': 'scaleZ'
}
LONG_TO_SHORT = {v: k for k, v in SHORT_TO_LONG.items()}

ROTATION_ATTRS = {'rotateX', 'rotateY', 'rotateZ', 'rx', 'ry', 'rz'}
SCALE_ATTRS = {'scaleX', 'scaleY', 'scaleZ', 'sx', 'sy', 'sz'}
TRANSLATE_ATTRS = {'translateX', 'translateY', 'translateZ', 'tx', 'ty', 'tz'}
TRANSFORM_ATTRS = TRANSLATE_ATTRS | ROTATION_ATTRS | SCALE_ATTRS


# =============================================================================
# Decorators
# =============================================================================

def viewport_off(func):
    """
    Decorator - turn off Maya display while func is running.
    If func fails, the error will be raised after viewport is restored.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        parallel = False
        if 'parallel' in cmds.evaluationManager(q=True, mode=True):
            cmds.evaluationManager(mode='off')
            parallel = True

        # Turn $gMainPane Off:
        mel.eval("paneLayout -e -manage false $gMainPane")
        cmds.refresh(suspend=True)
        # Hide the timeslider
        mel.eval("setTimeSliderVisible 0;")

        try:
            return func(*args, **kwargs)
        except Exception:
            raise
        finally:
            cmds.refresh(suspend=False)
            mel.eval("setTimeSliderVisible 1;")
            if parallel:
                cmds.evaluationManager(mode='parallel')
            mel.eval("paneLayout -e -manage true $gMainPane")

    return wrap


def undo(func):
    """Decorator - open/close undo chunk"""
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(openChunk=True)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise
        finally:
            cmds.undoInfo(closeChunk=True)

    return wrap


def noUndo(func):
    """Decorator - disable undo for this function"""
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(stateWithoutFlush=False)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise
        finally:
            cmds.undoInfo(stateWithoutFlush=True)

    return wrap


# =============================================================================
# Layer Stack Data Structures
# =============================================================================

@dataclass
class Contribution:
    """A single layer's contribution to an attribute."""
    layer: str
    curve_fn: Optional[anim.MFnAnimCurve] = None
    static_value: float = 0.0
    curve_name: str = ""


@dataclass
class LayerInfo:
    """Information about a single animation layer."""
    name: str
    weight_curve_fn: Optional[anim.MFnAnimCurve] = None
    weight_static: float = 1.0
    muted: bool = False
    rotation_mode: int = 0  # 0=component, 1=quaternion
    scale_mode: int = 0  # 0=additive, 1=multiplicative


@dataclass
class LayerStack:
    """
    Discovered layer structure for a node, used during override-to-additive conversion.
    """
    node: str
    target_layer: str
    layers: dict = field(default_factory=dict)
    layer_order: list = field(default_factory=list)
    contributions: dict = field(default_factory=dict)
    target_curves: dict = field(default_factory=dict)

    @classmethod
    def build(cls, node: str, target_layer: str) -> 'LayerStack':
        """Build a LayerStack by traversing blend node chains for all layer attributes."""
        stack = cls(node=node, target_layer=target_layer)

        # Get target layer properties
        stack.layers[target_layer] = LayerInfo(
            name=target_layer,
            weight_curve_fn=_get_weight_curve_fn(target_layer),
            weight_static=_get_static_weight(target_layer),
            muted=cmds.getAttr(f"{target_layer}.mute") if cmds.objExists(f"{target_layer}.mute") else False,
            rotation_mode=cmds.getAttr(f"{target_layer}.rotationAccumulationMode"),
            scale_mode=cmds.getAttr(f"{target_layer}.scaleAccumulationMode"),
        )

        # Build curve -> layer lookup
        curve_to_layer = {}
        all_layers = cmds.ls(type='animLayer') or []
        for layer in all_layers:
            layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []
            for curve in layer_curves:
                curve_to_layer[curve] = layer
            if layer != target_layer and layer not in stack.layers:
                stack.layers[layer] = LayerInfo(
                    name=layer,
                    weight_curve_fn=_get_weight_curve_fn(layer),
                    weight_static=_get_static_weight(layer),
                    muted=cmds.getAttr(f"{layer}.mute") if cmds.objExists(f"{layer}.mute") else False,
                    rotation_mode=cmds.getAttr(f"{layer}.rotationAccumulationMode") if cmds.objExists(f"{layer}.rotationAccumulationMode") else 0,
                    scale_mode=cmds.getAttr(f"{layer}.scaleAccumulationMode") if cmds.objExists(f"{layer}.scaleAccumulationMode") else 0,
                )

        layer_curves = cmds.animLayer(target_layer, query=True, animCurves=True) or []
        target_curve_set = set(layer_curves)

        layer_attrs = cmds.animLayer(target_layer, query=True, attribute=True) or []
        node_attrs = [a.split('.')[-1] for a in layer_attrs if a.startswith(f"{node}.")]

        if not node_attrs:
            return stack

        for attr in node_attrs:
            attr_long = normalizeAttrName(attr)
            target_curve_fn = _find_target_curve(node, attr_long, target_layer, target_curve_set)
            if target_curve_fn:
                stack.target_curves[attr_long] = target_curve_fn

            contribs = _traverse_for_contributions(node, attr_long, target_layer, target_curve_set, stack, curve_to_layer)
            if contribs:
                stack.contributions[attr_long] = contribs

        seen_layers = set()
        for attr, contribs in stack.contributions.items():
            for c in contribs:
                if c.layer and c.layer != target_layer and c.layer not in seen_layers:
                    seen_layers.add(c.layer)
                    stack.layer_order.append(c.layer)

        return stack

    def get_base_value(self, attr: str, time: float) -> float:
        """Get composited base value for an attribute at a single time."""
        attr_long = normalizeAttrName(attr)
        contribs = self.contributions.get(attr_long, [])

        if not contribs:
            if attr_long in SCALE_ATTRS and self.layers[self.target_layer].scale_mode == 1:
                return 1.0
            return 0.0

        is_rotation = attr_long in ROTATION_ATTRS
        mtime = api.MTime(time, api.MTime.uiUnit())

        total = 0.0
        for contrib in contribs:
            weight = self.get_layer_weight(contrib.layer, time)
            if contrib.curve_fn:
                value = contrib.curve_fn.evaluate(mtime)
                if is_rotation:
                    value = math.degrees(value)
            else:
                value = contrib.static_value
            total += value * weight

        return total

    def get_base_values(self, attr: str, times: list) -> list:
        """Get composited base values for an attribute at multiple times."""
        return [self.get_base_value(attr, t) for t in times]

    def get_layer_weight(self, layer: str, time: float) -> float:
        """Get a layer's weight at a specific time."""
        if layer == 'BaseAnimation' or layer is None:
            return 1.0

        layer_info = self.layers.get(layer)
        if not layer_info:
            if cmds.objExists(f"{layer}.weight"):
                return cmds.getAttr(f"{layer}.weight")
            return 1.0

        if layer_info.weight_curve_fn:
            mtime = api.MTime(time, api.MTime.uiUnit())
            return layer_info.weight_curve_fn.evaluate(mtime)

        return layer_info.weight_static

    def has_contribution(self, attr: str) -> bool:
        """Check if any layer below target contributes to this attr."""
        attr_long = normalizeAttrName(attr)
        return attr_long in self.contributions and len(self.contributions[attr_long]) > 0

    def diagnose(self) -> str:
        """Generate a diagnostic string showing the discovered layer structure."""
        lines = [
            f"LayerStack for '{self.node}' (target: {self.target_layer})",
            "=" * 60,
            f"Layer order (bottom to top): {self.layer_order}",
            "",
            f"Target curves ({len(self.target_curves)}):",
        ]

        for attr, curve_fn in sorted(self.target_curves.items()):
            num_keys = curve_fn.numKeys if curve_fn else 0
            lines.append(f"  {attr}: {num_keys} keys")

        lines.append("")
        lines.append("Contributions by attribute:")

        for attr, contribs in sorted(self.contributions.items()):
            lines.append(f"  {attr}:")
            for c in contribs:
                if c.curve_fn:
                    lines.append(f"    - {c.layer}: curve '{c.curve_name}' ({c.curve_fn.numKeys} keys)")
                else:
                    lines.append(f"    - {c.layer}: static = {c.static_value}")

        return "\n".join(lines)


# =============================================================================
# Helper Functions for LayerStack
# =============================================================================

def _get_weight_curve_fn(layer: str) -> Optional[anim.MFnAnimCurve]:
    """Get MFnAnimCurve for a layer's weight if animated."""
    weight_plug = f"{layer}.weight"
    if not cmds.objExists(weight_plug):
        return None

    curves = cmds.listConnections(weight_plug, source=True, destination=False, type='animCurve')
    if curves:
        sel = api.MSelectionList()
        sel.add(curves[0])
        return anim.MFnAnimCurve(sel.getDependNode(0))
    return None


def _get_static_weight(layer: str) -> float:
    """Get static weight value for a layer."""
    weight_plug = f"{layer}.weight"
    if cmds.objExists(weight_plug):
        return cmds.getAttr(weight_plug)
    return 1.0


def _find_target_curve(node: str, attr: str, target_layer: str,
                       target_curve_set: set) -> Optional[anim.MFnAnimCurve]:
    """Find the target layer's curve for a specific attribute."""
    for curve_name in target_curve_set:
        outputs = cmds.listConnections(f"{curve_name}.output", source=False,
                                        destination=True, plugs=True) or []
        for out in outputs:
            target = _trace_to_attribute(out)
            if target and target[0] == node and normalizeAttrName(target[1]) == attr:
                sel = api.MSelectionList()
                sel.add(curve_name)
                return anim.MFnAnimCurve(sel.getDependNode(0))
    return None


def _trace_to_attribute(plug: str, max_depth: int = 10) -> Optional[tuple]:
    """Trace from a plug through blend nodes to find the final node.attr."""
    visited = set()
    current = plug

    for _ in range(max_depth):
        if not current or current in visited:
            break
        visited.add(current)

        node = current.split('.')[0]
        node_type = cmds.nodeType(node)

        if node_type in ('transform', 'joint', 'ikHandle') or cmds.objectType(node, isAType='transform'):
            attr = current.split('.')[-1]
            return (node, attr)

        if 'Blend' in node_type or 'blend' in node_type:
            out_attr = 'output'
            if 'X' in current.split('.')[-1]:
                out_attr = 'outputX'
            elif 'Y' in current.split('.')[-1]:
                out_attr = 'outputY'
            elif 'Z' in current.split('.')[-1]:
                out_attr = 'outputZ'

            out_plug = f"{node}.{out_attr}"
            if cmds.objExists(out_plug):
                outputs = cmds.listConnections(out_plug, source=False,
                                               destination=True, plugs=True) or []
                if outputs:
                    current = outputs[0]
                    continue

        break

    return None


def _traverse_for_contributions(node: str, attr: str, target_layer: str,
                                target_curve_set: set, stack: 'LayerStack',
                                curve_to_layer: dict) -> list:
    """Traverse blend node chain to find all contributions below the target layer."""
    contributions = []
    plug = f"{node}.{attr}"

    conns = cmds.listConnections(plug, source=True, destination=False,
                                  plugs=True, skipConversionNodes=True) or []

    blend_node = None
    for conn in conns:
        conn_node = conn.split('.')[0]
        conn_type = cmds.nodeType(conn_node)

        if conn_type == 'pairBlend':
            return []

        if 'animBlend' in conn_type or conn_type == 'animBlendNodeBase':
            blend_node = conn_node
            break

    if not blend_node:
        return contributions

    visited = set()
    current_blend = blend_node

    while current_blend and current_blend not in visited:
        visited.add(current_blend)

        input_b_name = _get_input_b_name(current_blend, attr)
        input_b_plug = f"{current_blend}.{input_b_name}"

        input_b_conns = cmds.listConnections(input_b_plug, source=True, destination=False,
                                              plugs=True, skipConversionNodes=True) or []

        curve_fn = None
        static_value = 0.0
        source_layer = None
        curve_name = ""
        is_target_layer = False

        if input_b_conns:
            for conn in input_b_conns:
                conn_node = conn.split('.')[0]
                conn_type = cmds.nodeType(conn_node)

                if conn_type.startswith('animCurve'):
                    curve_name = conn_node

                    if conn_node in target_curve_set:
                        is_target_layer = True
                        break

                    sel = api.MSelectionList()
                    sel.add(conn_node)
                    curve_fn = anim.MFnAnimCurve(sel.getDependNode(0))
                    source_layer = curve_to_layer.get(conn_node)
                    break
        else:
            static_value = cmds.getAttr(input_b_plug)

        if not is_target_layer and (curve_fn or input_b_conns == []):
            contributions.append(Contribution(
                layer=source_layer or 'BaseAnimation',
                curve_fn=curve_fn,
                static_value=static_value,
                curve_name=curve_name,
            ))

        input_a_name = _get_input_a_name(current_blend, attr)
        input_a_plug = f"{current_blend}.{input_a_name}"

        input_a_conns = cmds.listConnections(input_a_plug, source=True, destination=False,
                                              skipConversionNodes=True) or []

        current_blend = None
        for conn in input_a_conns:
            conn_type = cmds.nodeType(conn)
            if 'animBlend' in conn_type or conn_type == 'animBlendNodeBase':
                current_blend = conn
                break
            elif conn_type.startswith('animCurve'):
                layer_name = curve_to_layer.get(conn)
                is_base = not layer_name or layer_name == 'BaseAnimation'

                if is_base:
                    sel = api.MSelectionList()
                    sel.add(conn)
                    base_curve_fn = anim.MFnAnimCurve(sel.getDependNode(0))
                    contributions.append(Contribution(
                        layer='BaseAnimation',
                        curve_fn=base_curve_fn,
                        static_value=0.0,
                        curve_name=conn,
                    ))
                break

    contributions.reverse()
    return contributions


def _get_input_a_name(blend_node: str, attr: str) -> str:
    """Get correct inputA plug name based on blend node type and attribute."""
    blend_type = cmds.nodeType(blend_node)
    attr_long = normalizeAttrName(attr)

    if blend_type == "animBlendNodeAdditiveRotation":
        if attr_long == 'rotateX':
            return "inputAX"
        elif attr_long == 'rotateY':
            return "inputAY"
        elif attr_long == 'rotateZ':
            return "inputAZ"

    return "inputA"


def _get_input_b_name(blend_node: str, attr: str) -> str:
    """Get correct inputB plug name based on blend node type and attribute."""
    blend_type = cmds.nodeType(blend_node)
    attr_long = normalizeAttrName(attr)

    if blend_type == "animBlendNodeAdditiveRotation":
        if attr_long == 'rotateX':
            return "inputBX"
        elif attr_long == 'rotateY':
            return "inputBY"
        elif attr_long == 'rotateZ':
            return "inputBZ"

    return "inputB"


# =============================================================================
# Attribute Utilities
# =============================================================================

def normalizeAttrName(attr):
    """Convert attribute name to long form."""
    return SHORT_TO_LONG.get(attr, attr)


def getAttrAxis(attr):
    """Get the axis (X, Y, Z) from an attribute name."""
    attr_long = normalizeAttrName(attr)
    if attr_long.endswith('X'):
        return 'X'
    elif attr_long.endswith('Y'):
        return 'Y'
    elif attr_long.endswith('Z'):
        return 'Z'
    return None


# =============================================================================
# Pin Management Functions
# =============================================================================

def get_pin_groups():
    """Get all pin groups under the master group."""
    global MASTER_GROUP
    found_pin_groups = []
    if cmds.objExists(MASTER_GROUP):
        master_group_children = cmds.listRelatives(MASTER_GROUP, type='transform') or []
        for found_pin_group in master_group_children:
            found_pin_groups.append(found_pin_group)
    return found_pin_groups


def get_pins(pin_groups=None):
    """Get all pins in the specified pin groups."""
    if isinstance(pin_groups, str):
        pin_groups = [pin_groups]
    found_pins = []
    if not pin_groups:
        pin_groups = get_pin_groups()
    for group in pin_groups:
        children = cmds.listRelatives(group, type='transform') or []
        for child in children:
            if cmds.attributeQuery('control', node=child, exists=True):
                found_pins.append(child)
            else:
                grandchildren = cmds.listRelatives(child, type='transform') or []
                for grandchild in grandchildren:
                    if cmds.attributeQuery('control', node=grandchild, exists=True):
                        found_pins.append(grandchild)
    return found_pins


def get_master_group(group_override=None):
    """Get or create the master pin group."""
    global MASTER_GROUP
    if isinstance(group_override, str):
        MASTER_GROUP = group_override
    if not cmds.objExists(MASTER_GROUP):
        MASTER_GROUP = cmds.createNode('transform',
                                       name=MASTER_GROUP,
                                       skipSelect=True)
    return MASTER_GROUP


def create_new_pin_group():
    """Create a new pin group under the master group."""
    master_group = get_master_group()
    new_pin_group = cmds.createNode('transform',
                                    name=PIN_GROUP,
                                    parent=master_group,
                                    skipSelect=True)
    return new_pin_group


def get_selectionList(selection):
    """Convert selection to MSelectionList."""
    if not selection:
        return api.MGlobal.getActiveSelectionList()
    elif isinstance(selection, str):
        return api.MGlobal.getSelectionListByName(selection)
    elif isinstance(selection, list):
        nodes = api.MSelectionList()
        for sel in selection:
            try:
                node = api.MGlobal.getSelectionListByName(sel)
                nodes.merge(node)
            except:
                api.MGlobal.displayError(
                    "Could not fetch selection. "
                    "Try submitting an MSelectionList "
                    "or a list of string names."
                )
        return nodes
    else:
        return selection


def create_locator_pin(control_data, pin_group):
    """Create a locator pin with control data attributes.

    The control name stored in control_data['control'] may be a long DAG path
    (e.g., '|group1|arm_ctrl'). We extract the short name for the locator
    naming (since Maya node names cannot contain pipes), but store the full
    long name in the 'control' attribute for unique identification.
    """
    control_long_name = control_data['control']
    # Extract short name for locator naming (Maya doesn't allow pipes in names)
    control_short_name = get_short_name(control_long_name)

    locator = cmds.spaceLocator(name=control_short_name + AP_SUFFIX)[0]
    cmds.setAttr(locator + ".scale", *[LOCATOR_SCALE] * 3)
    # Parent returns the new long path - use it to avoid ambiguity with duplicate names
    locator = cmds.parent(locator, pin_group)[0]

    for attr in ['x', 'y', 'z']:
        cmds.setAttr(locator + '.s' + attr, keyable=False, channelBox=True)

    for key, value in PIN_DATA.items():
        cmds.addAttr(locator, longName=key, **value)

    for key, value in control_data.items():
        if isinstance(value, (str, list)):
            if isinstance(value, list):
                value = ' '.join(map(str, value))
            cmds.setAttr(locator + '.' + key, value, type='string')
        else:
            cmds.setAttr(locator + '.' + key, value)

    return locator


# =============================================================================
# Validation Functions
# =============================================================================

def validate_selection(sel_list):
    """Validate and filter selection for valid transform nodes."""
    current_pins = get_pins()
    validated_sel_list = api.MSelectionList()

    # Build a set of long names for controls that are already pinned
    pinned_control_long_names = set()
    pin_long_names = set()
    for pin in current_pins:
        # Get the stored control long name from the pin's attribute
        if cmds.attributeQuery('control', node=pin, exists=True):
            stored_control = cmds.getAttr(pin + '.control')
            if stored_control:
                pinned_control_long_names.add(stored_control)
        # Also get the pin's own long name
        pin_long_names.add(cmds.ls(pin, long=True)[0])

    for i in range(sel_list.length()):
        try:
            dag = sel_list.getDagPath(i)
        except TypeError:
            continue

        control_dep = sel_list.getDependNode(i)
        controlFN = api.MFnDependencyNode(control_dep)
        # Use long name for unique identification
        control_long_name = dag.fullPathName()
        control_short_name = get_short_name(control_long_name)

        # Check if this control is already pinned (using long name comparison)
        if control_long_name in pinned_control_long_names:
            api.MGlobal.displayError(
                "Node '%s' is already pinned! Skipping..." % control_short_name)
            continue

        # Check if user selected a pin itself (using long name comparison)
        if control_long_name in pin_long_names:
            api.MGlobal.displayError(
                "Node '%s' is a pin! Skipping..." % control_short_name)
            continue

        if controlFN.typeName not in ['transform', 'joint', 'ikHandle']:
            api.MGlobal.displayError(
                "Node '%s' is not a valid transform node. Skipping..." % control_short_name)
            continue

        if is_locked_or_not_keyable(controlFN, 'translate') and \
           is_locked_or_not_keyable(controlFN, 'rotate'):
            api.MGlobal.displayError(
                "Node '%s' has no available transform channels. Skipping..." % control_short_name)
            continue

        validated_sel_list.add(sel_list.getDependNode(i))

    return validated_sel_list


def validate_framerange(start_frame, end_frame):
    """Validate and return frame range."""
    if start_frame is None:
        start_frame = cmds.playbackOptions(query=True, minTime=True)

    if end_frame is None:
        end_frame = cmds.playbackOptions(query=True, maxTime=True)

    if start_frame > end_frame:
        api.MGlobal.displayError("Start frame needs to be before end frame!")
        return None, None

    return start_frame, end_frame


def validate_bakerange(pins_to_bake, start_frame, end_frame):
    """Validate bake range against pin ranges."""
    all_pins_start_frame = set()
    all_pins_end_frame = set()
    for pin in pins_to_bake:
        pin_start = cmds.getAttr(pin + '.start_frame')
        pin_end = cmds.getAttr(pin + '.end_frame')
        all_pins_start_frame.add(pin_start)
        all_pins_end_frame.add(pin_end)
    all_pins_start_frame = list(all_pins_start_frame)[0]
    all_pins_end_frame = list(all_pins_end_frame)[0]

    selected_range = str(cmds.timeControl(
        aPlayBackSliderPython,
        q=True,
        range=True))
    selected_range = [float(x) for x in selected_range.strip('"').split(':')]
    if not selected_range[1] - 1 == selected_range[0]:
        start_frame, end_frame = selected_range

    start_frame, end_frame = validate_framerange(start_frame, end_frame)

    if start_frame < all_pins_start_frame:
        start_frame = all_pins_start_frame

    if end_frame > all_pins_end_frame:
        end_frame = all_pins_end_frame

    return start_frame, end_frame


def is_locked_or_not_keyable(controlFN, attribute):
    """Check if an attribute is locked or not keyable."""
    plug = controlFN.findPlug(attribute, False)
    if any(plug.child(p).isLocked for p in range(plug.numChildren())):
        return True
    if not any(plug.child(p).isKeyable for p in range(plug.numChildren())):
        return True
    for p in range(plug.numChildren()):
        plug_array = plug.child(p).connectedTo(True, False)
        if plug_array:
            if plug_array[0].isChild:
                # Check if the connected node is an animation blend node (from anim layers)
                # Animation layer connections should not be treated as "locked"
                conn_node = api.MFnDependencyNode(plug_array[0].node())
                conn_type = conn_node.typeName
                if 'animBlend' in conn_type:
                    continue  # Skip animation layer connections
                return True
    return False


# =============================================================================
# Key and Curve Functions
# =============================================================================

def get_keys_from_obj_attribute(controlFN, attribute):
    """Get keyframe times from an attribute."""
    keys = set()
    attribute_plug = controlFN.findPlug(attribute, False)
    if attribute_plug.isCompound:
        for c in range(attribute_plug.numChildren()):
            plug = attribute_plug.child(c)
            if anim.MAnimUtil.isAnimated(plug):
                keys.update(get_keys_from_curve(plug))
    else:
        if anim.MAnimUtil.isAnimated(attribute_plug):
            keys.update(get_keys_from_curve(attribute_plug))
    return list(keys)


def get_keys_from_curve(plug):
    """Get keyframe times from an animation curve."""
    curve = anim.MFnAnimCurve(plug)
    return [curve.input(k).value for k in range(curve.numKeys)]


def bookend_curves(controls, start_frame, end_frame):
    """
    Insert bookend keys one frame before/after the work area to preserve curve shape.
    """
    attrs = ['translateX', 'translateY', 'translateZ',
             'rotateX', 'rotateY', 'rotateZ']

    for control in controls:
        for attr in attrs:
            plug = f'{control}.{attr}'

            if not cmds.objExists(plug):
                continue
            if not cmds.getAttr(plug, keyable=True):
                continue

            key_times = cmds.keyframe(plug, query=True, timeChange=True) or []

            if not key_times:
                continue

            keys_before = [t for t in key_times if t < start_frame]
            if keys_before:
                bookend_time = start_frame - 1
                if bookend_time not in key_times:
                    cmds.setKeyframe(plug, time=bookend_time, insert=True)

            keys_after = [t for t in key_times if t > end_frame]
            if keys_after:
                bookend_time = end_frame + 1
                if bookend_time not in key_times:
                    cmds.setKeyframe(plug, time=bookend_time, insert=True)


def get_long_name(mobject):
    """Get the full DAG path (long name) for a Maya object.

    This ensures unique identification even when multiple objects
    share the same short name.
    """
    try:
        # Use MFnDagNode to get the full path directly
        dag_fn = api.MFnDagNode(mobject)
        return dag_fn.fullPathName()
    except (TypeError, RuntimeError):
        # Fall back to dependency node name if not a DAG node
        fn = api.MFnDependencyNode(mobject)
        return fn.name()


def get_short_name(long_name):
    """Extract the short name from a long DAG path.

    Example: '|group1|arm_ctrl' -> 'arm_ctrl'
    Also handles namespaces: '|group1|ns:arm_ctrl' -> 'arm_ctrl'
    """
    short = long_name.split('|')[-1]  # Get last part of path
    short = short.split(':')[-1]  # Remove namespace
    return short


def read_control_data(control, start_frame, end_frame):
    """Read control data for pin creation."""
    control_data = OrderedDict()
    controlFN = api.MFnDependencyNode(control)
    # Use long name (full DAG path) to uniquely identify the control
    control_name = get_long_name(control)

    t_keys = get_keys_from_obj_attribute(controlFN, 'translate')
    r_keys = get_keys_from_obj_attribute(controlFN, 'rotate')

    t_lock = is_locked_or_not_keyable(controlFN, 'translate')
    r_lock = is_locked_or_not_keyable(controlFN, 'rotate')

    bp_keys = []
    if 'blendParent1' in cmds.listAttr(control_name):
        bp_key_times = get_keys_from_obj_attribute(controlFN, 'blendParent1')
        if bp_key_times:
            key_values = []
            for time in bp_key_times:
                key_value = cmds.getAttr(control_name + '.blendParent1', time=time)
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
    control_data['preserve_blendParent'] = bp_keys

    return control_data


# =============================================================================
# Baking Functions
# =============================================================================

@viewport_off
def do_bake(nodes_to_bake, start_frame, end_frame, sample=1, destinationLayer=None):
    """Bake animation to nodes."""
    try:
        bake_kwargs = {
            'simulation': True,
            'time': (start_frame, end_frame),
            'sampleBy': sample,
            'oversamplingRate': 1,
            'disableImplicitControl': True,
            'preserveOutsideKeys': True,
            'sparseAnimCurveBake': False,
            'removeBakedAttributeFromLayer': False,
            'removeBakedAnimFromLayer': False,
            'bakeOnOverrideLayer': False,
            'minimizeRotation': True,
            'controlPoints': False,
            'shape': True
        }

        if destinationLayer:
            bake_kwargs['at'] = ("tx", "ty", "tz", "rx", "ry", "rz")
            bake_kwargs['destinationLayer'] = destinationLayer
        else:
            bake_kwargs['at'] = ("tx", "ty", "tz", "rx", "ry", "rz", "blendParent1")

        cmds.bakeResults(nodes_to_bake, **bake_kwargs)
        return True
    except:
        return False


@viewport_off
def do_bake_to_layer(controls_to_bake, start_frame, end_frame, sample=1,
                     unrollRotations=True, debug=False, constraints=None):
    """
    Hybrid bake approach: Bake to override layer, then convert to additive.
    This is ~3x faster than frame-by-frame Python baking for animation layers.
    """
    original_time = cmds.currentTime(query=True)

    try:
        layer_name = 'AnimPin_Layer'
        counter = 1
        while cmds.objExists(layer_name):
            layer_name = f'AnimPin_Layer_{counter}'
            counter += 1

        layer = cmds.animLayer(layer_name, override=True)

        bake_kwargs = {
            'simulation': True,
            'time': (start_frame, end_frame),
            'sampleBy': sample,
            'oversamplingRate': 1,
            'disableImplicitControl': True,
            'preserveOutsideKeys': True,
            'sparseAnimCurveBake': False,
            'minimizeRotation': True,
            'destinationLayer': layer,
            'at': ("tx", "ty", "tz", "rx", "ry", "rz")
        }
        cmds.bakeResults(controls_to_bake, **bake_kwargs)

        if constraints:
            for constraint in constraints:
                if cmds.objExists(constraint):
                    cmds.delete(constraint)

        convertOverrideToAdditive(layer)

        if unrollRotations:
            eulerFilterLayer(layer, nodes=controls_to_bake)

        return True

    except Exception as e:
        api.MGlobal.displayError(f"Bake to layer failed: {e}")
        return False

    finally:
        cmds.currentTime(original_time, edit=True)


def match_keys_procedure(pins_to_bake, start_frame, end_frame, composite=True):
    """Match keys procedure after baking."""
    for pin in pins_to_bake:
        control = cmds.getAttr(pin + '.control')

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
            attribute='t',
            time=(min(translate_keys or [start_frame]), max(translate_keys or [end_frame])),
            query=True) or [])
        translate_keys_to_remove = list(set(translate_keys_baked - set(translate_keys)))

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
            attribute='r',
            time=(min(rotate_keys or [start_frame]), max(rotate_keys or [end_frame])),
            query=True) or [])
        rotate_keys_to_remove = list(set(rotate_keys_baked - set(rotate_keys)))

        keys_baked = list(translate_keys_baked | rotate_keys_baked)
        if composite:
            composited_keys = list(set(translate_keys + rotate_keys))
            keys_to_remove = list(set(keys_baked) - set(composited_keys))
            for key in keys_to_remove:
                cmds.cutKey(control, t=(key,), attribute=('t', 'r'), clear=True)

        keys_baked.insert(0, keys_baked[0] - 1)
        keys_baked.append(keys_baked[-1] + 1)
        keys = []
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        for match in tuple_rex.finditer(bp_keys):
            keys.append(float(match.group(0).split(', ')[0]))
        bp_keys_to_remove = list(set(keys_baked) - set(keys))
        for key in _to_ranges(bp_keys_to_remove):
            cmds.cutKey(control, time=key, attribute=('blendParent1'), clear=True)

    return True


def _to_ranges(iterable):
    """Convert list of integers to ranges."""
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


# =============================================================================
# Layer Detection Functions
# =============================================================================

def control_has_anim_layers(control):
    """Check if a control has any attributes on animation layers (excluding BaseAnimation).

    Uses long names (full DAG paths) for comparison to correctly handle
    multiple objects with the same short name.
    """
    all_layers = cmds.ls(type='animLayer') or []

    # Normalize control to long name for accurate comparison
    control_long_names = cmds.ls(control, long=True) or []
    if not control_long_names:
        return False
    control_long = control_long_names[0]

    for layer in all_layers:
        if layer == 'BaseAnimation':
            continue

        layer_plugs = cmds.animLayer(layer, query=True, layeredPlug=True) or []
        for plug in layer_plugs:
            plug_node = plug.split('.')[0]
            # Convert to long name for comparison
            plug_node_long_names = cmds.ls(plug_node, long=True) or []
            if plug_node_long_names and plug_node_long_names[0] == control_long:
                return True

        layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []
        for curve in layer_curves:
            connections = cmds.listConnections(curve + '.output', plugs=True) or []
            for conn in connections:
                conn_node = conn.split('.')[0]
                # Convert to long name for comparison
                conn_node_long_names = cmds.ls(conn_node, long=True) or []
                if conn_node_long_names and conn_node_long_names[0] == control_long:
                    return True

                node_type = cmds.nodeType(conn_node)
                if 'animBlend' in node_type:
                    blend_outputs = cmds.listConnections(conn_node + '.output', plugs=True) or []
                    for bout in blend_outputs:
                        bout_node = bout.split('.')[0]
                        # Convert to long name for comparison
                        bout_long_names = cmds.ls(bout_node, long=True) or []
                        if bout_long_names and bout_long_names[0] == control_long:
                            return True

    return False


def find_pin_for_control(control, all_pin_groups=None):
    """Find the pin locator and pin_group for a given control.

    Args:
        control: Control name (short or long) to find pin for
        all_pin_groups: Optional list of pin groups to search

    Returns:
        Tuple of (pin, pin_group) if found, or (None, None) if not found
    """
    if all_pin_groups is None:
        all_pin_groups = get_pin_groups()

    all_pins = get_pins(all_pin_groups)

    # Normalize input control name to long name for comparison
    control_long_names = cmds.ls(control, long=True) or []
    if not control_long_names:
        return (None, None)
    control_long_name = control_long_names[0]

    for pin in all_pins:
        pin_control = cmds.getAttr(pin + '.control')
        # Compare using long names for unique identification
        if pin_control == control_long_name:
            parent = cmds.listRelatives(pin, parent=True)
            while parent:
                parent = parent[0]
                if parent in all_pin_groups:
                    return (pin, parent)
                parent = cmds.listRelatives(parent, parent=True)

    return (None, None)


def find_pin_groups_from_selection(selection):
    """Find pin groups from selection."""
    all_pin_groups = get_pin_groups()
    pin_group_list = set()

    for sel in selection:
        if sel in all_pin_groups:
            pin_group_list.add(sel)
        elif cmds.attributeQuery('control', node=sel, exists=True):
            parent = cmds.listRelatives(sel, parent=True)
            while parent:
                parent = parent[0]
                if parent in all_pin_groups:
                    pin_group_list.add(parent)
                    break
                parent = cmds.listRelatives(parent, parent=True)
        else:
            pin, pin_group = find_pin_for_control(sel, all_pin_groups)
            if pin_group:
                pin_group_list.add(pin_group)
                continue

            for pg in all_pin_groups:
                constraints = cmds.listRelatives(pg, type='parentConstraint') or []
                for constraint in constraints:
                    targets = cmds.parentConstraint(constraint, query=True, targetList=True) or []
                    if sel in targets:
                        pin_group_list.add(pg)
                        break

    return pin_group_list


# =============================================================================
# Constraint Functions
# =============================================================================

def parent_constraint_with_skips(driver, driven):
    """Create a parent constraint from the driver to the driven, skipping locked axes."""
    skipT = []
    skipR = []

    for axis in ['X', 'Y', 'Z']:
        t_attr = f"{driven}.translate{axis}"
        r_attr = f"{driven}.rotate{axis}"

        if cmds.getAttr(t_attr, lock=True):
            skipT.append(axis.lower())
        if cmds.getAttr(r_attr, lock=True):
            skipR.append(axis.lower())

    constraint = ""
    try:
        constraint = cmds.parentConstraint(
            driver,
            driven,
            mo=False,
            skipTranslate=skipT,
            skipRotate=skipR
        )[0] or ""
    except Exception as e:
        log.warning(f"Could not apply parentConstraint: {e}")

    return constraint


# =============================================================================
# Layer Conversion Main Function
# =============================================================================

def convertOverrideToAdditive(layer, debug=False, verify=False, tolerance=0.01):
    """
    Convert an override animation layer to an additive layer.

    This modifies the layer's animation curves in place, replacing absolute
    values with delta values relative to the layer stack below.
    """
    if not cmds.objExists(layer):
        raise ValueError(f"Layer '{layer}' does not exist")

    if cmds.nodeType(layer) != 'animLayer':
        raise ValueError(f"'{layer}' is not an animation layer")

    is_valid, error_msg = validateLayerState(layer)
    if not is_valid:
        raise ValueError(error_msg)

    is_override = cmds.getAttr(f"{layer}.override")
    if not is_override:
        log.warning(f"Layer '{layer}' is already additive, nothing to convert")
        return {'curves_converted': 0, 'keys_processed': 0, 'nodes_affected': []}

    layer_attrs = cmds.animLayer(layer, query=True, attribute=True) or []
    nodes = list(set(a.split('.')[0] for a in layer_attrs if '.' in a))

    if not nodes:
        log.warning(f"No nodes found on layer '{layer}'")
        return {'curves_converted': 0, 'keys_processed': 0, 'nodes_affected': []}

    stats = {
        'curves_converted': 0,
        'keys_processed': 0,
        'nodes_affected': [],
    }

    cmds.undoInfo(openChunk=True, chunkName=f"Convert {layer} to Additive")
    try:
        for node in nodes:
            try:
                stack = LayerStack.build(node, layer)

                if not stack.target_curves:
                    continue

                layer_info = stack.layers[layer]
                rotation_mode = layer_info.rotation_mode

                rot_attrs = {'rotateX', 'rotateY', 'rotateZ'}
                has_all_rotations = rot_attrs.issubset(set(stack.target_curves.keys()))

                if rotation_mode == 1 and has_all_rotations:
                    keys_converted = _convertRotationsWithStack(node, stack)
                    stats['curves_converted'] += 3
                    stats['keys_processed'] += keys_converted
                    processed_attrs = rot_attrs
                else:
                    processed_attrs = set()

                for attr, curve_fn in stack.target_curves.items():
                    if attr in processed_attrs:
                        continue

                    keys_converted = _convertCurveWithStack(attr, curve_fn, stack)
                    stats['curves_converted'] += 1
                    stats['keys_processed'] += keys_converted

                stats['nodes_affected'].append(node)

            except Exception as e:
                log.error(f"Error converting curves for '{node}': {e}")

        cmds.setAttr(f"{layer}.override", 0)

    finally:
        cmds.undoInfo(closeChunk=True)

    return stats


def validateLayerState(layer):
    """Validate that the layer can be converted."""
    if not cmds.objExists(layer):
        return False, f"Layer '{layer}' does not exist"

    if cmds.getAttr(f"{layer}.mute"):
        return False, f"Layer '{layer}' is muted. Unmute it before conversion."

    all_layers = cmds.ls(type='animLayer')
    for other_layer in all_layers:
        if other_layer != layer and cmds.objExists(f"{other_layer}.solo"):
            if cmds.getAttr(f"{other_layer}.solo"):
                if not cmds.getAttr(f"{layer}.solo"):
                    return False, f"Layer '{layer}' is muted because '{other_layer}' is soloed."

    return True, None


def _convertCurveWithStack(attr: str, curve_fn: anim.MFnAnimCurve, stack: 'LayerStack') -> int:
    """Convert a single curve using LayerStack for base values."""
    num_keys = curve_fn.numKeys
    if num_keys == 0:
        return 0

    curve_name = api.MFnDependencyNode(curve_fn.object()).name()

    key_times = []
    for i in range(num_keys):
        mtime = curve_fn.input(i)
        key_times.append(mtime.asUnits(api.MTime.uiUnit()))

    base_values = stack.get_base_values(attr, key_times)

    attr_long = normalizeAttrName(attr)
    is_rotation = attr_long in ROTATION_ATTRS
    is_scale = attr_long in SCALE_ATTRS

    layer_info = stack.layers[stack.target_layer]
    scale_mode = layer_info.scale_mode

    for i in range(num_keys):
        override_value = curve_fn.value(i)
        base_value = base_values[i]

        if is_rotation:
            override_deg = math.degrees(override_value)
            delta_deg = override_deg - base_value
            delta_value = delta_deg
        elif is_scale and scale_mode == 1:
            if abs(base_value) > 0.0001:
                delta_value = override_value / base_value
            else:
                delta_value = 1.0
        else:
            delta_value = override_value - base_value

        cmds.keyframe(curve_name, index=(i,), valueChange=delta_value, absolute=True)

    return num_keys


def _convertRotationsWithStack(node: str, stack: 'LayerStack') -> int:
    """Convert rotation curves using quaternion delta calculation."""
    curve_fns = {
        'rotateX': stack.target_curves['rotateX'],
        'rotateY': stack.target_curves['rotateY'],
        'rotateZ': stack.target_curves['rotateZ'],
    }

    curve_names = {
        attr: api.MFnDependencyNode(fn.object()).name()
        for attr, fn in curve_fns.items()
    }

    curve_x = curve_fns['rotateX']
    num_keys = curve_x.numKeys
    if num_keys == 0:
        return 0

    key_times = []
    for i in range(num_keys):
        mtime = curve_x.input(i)
        key_times.append(mtime.asUnits(api.MTime.uiUnit()))

    base_rx = stack.get_base_values('rotateX', key_times)
    base_ry = stack.get_base_values('rotateY', key_times)
    base_rz = stack.get_base_values('rotateZ', key_times)

    override_rx = [math.degrees(curve_fns['rotateX'].value(i)) for i in range(num_keys)]
    override_ry = [math.degrees(curve_fns['rotateY'].value(i)) for i in range(num_keys)]
    override_rz = [math.degrees(curve_fns['rotateZ'].value(i)) for i in range(num_keys)]

    delta_rx, delta_ry, delta_rz = computeQuaternionDelta(
        override_rx, override_ry, override_rz,
        base_rx, base_ry, base_rz
    )

    for i in range(num_keys):
        cmds.keyframe(curve_names['rotateX'], index=(i,), valueChange=delta_rx[i], absolute=True)
        cmds.keyframe(curve_names['rotateY'], index=(i,), valueChange=delta_ry[i], absolute=True)
        cmds.keyframe(curve_names['rotateZ'], index=(i,), valueChange=delta_rz[i], absolute=True)

    return num_keys * 3


# =============================================================================
# Quaternion Math
# =============================================================================

def computeQuaternionDelta(override_rx, override_ry, override_rz,
                           base_rx, base_ry, base_rz):
    """Compute rotation delta using quaternion math: delta = override * inverse(base)"""
    delta_rx = []
    delta_ry = []
    delta_rz = []

    for i in range(len(override_rx)):
        o_rx = math.radians(override_rx[i])
        o_ry = math.radians(override_ry[i])
        o_rz = math.radians(override_rz[i])
        b_rx = math.radians(base_rx[i])
        b_ry = math.radians(base_ry[i])
        b_rz = math.radians(base_rz[i])

        override_quat = eulerToQuaternion(o_rx, o_ry, o_rz)
        base_quat = eulerToQuaternion(b_rx, b_ry, b_rz)

        base_inv = quaternionInverse(base_quat)
        delta_quat = quaternionMultiply(override_quat, base_inv)

        d_rx, d_ry, d_rz = quaternionToEuler(delta_quat)

        delta_rx.append(math.degrees(d_rx))
        delta_ry.append(math.degrees(d_ry))
        delta_rz.append(math.degrees(d_rz))

    return delta_rx, delta_ry, delta_rz


def eulerToQuaternion(rx, ry, rz):
    """Convert XYZ euler angles (radians) to quaternion [w, x, y, z]."""
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return [w, x, y, z]


def quaternionInverse(q):
    """Compute inverse of a unit quaternion."""
    return [q[0], -q[1], -q[2], -q[3]]


def quaternionMultiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [w, x, y, z]


def quaternionPower(q, t):
    """
    Raise a quaternion to a power t.

    For unit quaternions, q^t interpolates between identity (t=0) and q (t=1).
    Used to "un-slerp" weighted rotations: if S = slerp(identity, D, w),
    then D = S^(1/w).

    Args:
        q: Quaternion as [w, x, y, z]
        t: Power to raise to

    Returns:
        Quaternion [w, x, y, z] representing q^t
    """
    w, x, y, z = q

    # Handle identity quaternion (no rotation)
    vec_len = math.sqrt(x*x + y*y + z*z)
    if vec_len < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]

    # Convert to angle-axis representation
    # q = cos(θ/2) + sin(θ/2) * axis
    # w = cos(θ/2), vec_len = sin(θ/2)
    half_angle = math.atan2(vec_len, w)

    # Scale the angle by power t
    new_half_angle = half_angle * t

    # Convert back to quaternion
    axis_x = x / vec_len
    axis_y = y / vec_len
    axis_z = z / vec_len

    new_w = math.cos(new_half_angle)
    new_sin = math.sin(new_half_angle)
    new_x = axis_x * new_sin
    new_y = axis_y * new_sin
    new_z = axis_z * new_sin

    return [new_w, new_x, new_y, new_z]


def quaternionToEuler(q):
    """Convert quaternion to XYZ euler angles (radians)."""
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        ry = math.copysign(math.pi / 2, sinp)
    else:
        ry = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)

    return rx, ry, rz


# =============================================================================
# Euler Filter
# =============================================================================

def eulerFilter(rot_x, rot_y, rot_z):
    """
    Euler filter using NumPy vectorization.
    Handles gimbal lock and rotation wrapping to prevent flips.
    """
    if len(rot_x) < 2:
        return rot_x, rot_y, rot_z

    rx = np.array(rot_x, dtype=np.float64)
    ry = np.array(rot_y, dtype=np.float64)
    rz = np.array(rot_z, dtype=np.float64)

    n = len(rx)

    filtered_x = np.empty(n, dtype=np.float64)
    filtered_y = np.empty(n, dtype=np.float64)
    filtered_z = np.empty(n, dtype=np.float64)

    filtered_x[0] = rx[0]
    filtered_y[0] = ry[0]
    filtered_z[0] = rz[0]

    fine_offsets = np.array([0, 360, -360], dtype=np.float64)
    ox, oy, oz = np.meshgrid(fine_offsets, fine_offsets, fine_offsets, indexing='ij')
    offset_combos = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)

    for i in range(1, n):
        prev = np.array([filtered_x[i - 1], filtered_y[i - 1], filtered_z[i - 1]])
        current = np.array([rx[i], ry[i], rz[i]])

        delta = current - prev
        n_rotations = np.round(delta / 360.0) * 360.0
        unwrapped = current - n_rotations

        candidates = unwrapped + offset_combos
        distances = np.abs(candidates - prev).sum(axis=1)
        best_idx = np.argmin(distances)
        best = candidates[best_idx]
        best_dist = distances[best_idx]

        if best_dist > 90:
            gimbal_variants = np.array([
                [current[0] + 180, 180 - current[1], current[2] + 180],
                [current[0] - 180, 180 - current[1], current[2] - 180],
                [current[0] + 180, -180 - current[1], current[2] + 180],
                [current[0] - 180, -180 - current[1], current[2] - 180],
            ], dtype=np.float64)

            for gv in gimbal_variants:
                g_delta = gv - prev
                g_n_rot = np.round(g_delta / 360.0) * 360.0
                g_unwrapped = gv - g_n_rot

                g_candidates = g_unwrapped + offset_combos
                g_distances = np.abs(g_candidates - prev).sum(axis=1)
                g_best_idx = np.argmin(g_distances)
                g_best_dist = g_distances[g_best_idx]

                if g_best_dist < best_dist:
                    best_dist = g_best_dist
                    best = g_candidates[g_best_idx]

        filtered_x[i] = best[0]
        filtered_y[i] = best[1]
        filtered_z[i] = best[2]

    return filtered_x.tolist(), filtered_y.tolist(), filtered_z.tolist()


def eulerFilterLayer(layer, nodes=None):
    """Apply euler filter to rotation curves on an animation layer."""
    if not cmds.objExists(layer):
        raise RuntimeError(f"Layer '{layer}' does not exist")

    layer_curves = getLayerAnimCurves(layer)

    if not layer_curves:
        return {}

    node_curves = {}
    for curve_node, (node, attr) in layer_curves.items():
        if nodes is not None:
            if isinstance(nodes, str):
                nodes = [nodes]
            if node not in nodes:
                continue

        if node not in node_curves:
            node_curves[node] = {}
        node_curves[node][attr] = curve_node

    processed = {}

    for node, curves in node_curves.items():
        rot_attrs = ['rotateX', 'rotateY', 'rotateZ']
        if not all(attr in curves for attr in rot_attrs):
            continue

        curve_fns = {}
        for attr in rot_attrs:
            curve_name = curves[attr]
            sel_list = api.MSelectionList()
            sel_list.add(curve_name)
            curve_fns[attr] = anim.MFnAnimCurve(sel_list.getDependNode(0))

        curve_x = curve_fns['rotateX']
        curve_y = curve_fns['rotateY']
        curve_z = curve_fns['rotateZ']

        num_keys = curve_x.numKeys
        if num_keys < 2:
            continue

        rot_x = []
        rot_y = []
        rot_z = []

        for i in range(num_keys):
            rot_x.append(math.degrees(curve_x.value(i)))
            rot_y.append(math.degrees(curve_y.value(i)))
            rot_z.append(math.degrees(curve_z.value(i)))

        filtered_x, filtered_y, filtered_z = eulerFilter(rot_x, rot_y, rot_z)

        changes_made = False
        for i in range(num_keys):
            if (abs(filtered_x[i] - rot_x[i]) > 0.001 or
                abs(filtered_y[i] - rot_y[i]) > 0.001 or
                abs(filtered_z[i] - rot_z[i]) > 0.001):
                changes_made = True
                break

        if not changes_made:
            continue

        for i in range(num_keys):
            curve_x.setValue(i, math.radians(filtered_x[i]))
            curve_y.setValue(i, math.radians(filtered_y[i]))
            curve_z.setValue(i, math.radians(filtered_z[i]))

        processed[node] = num_keys

    return processed


def getLayerAnimCurves(layer):
    """Get all animation curves on a layer and their associated node.attr pairs."""
    curves = {}
    layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []

    for curve in layer_curves:
        node_attr = getCurveTarget(curve)
        if node_attr:
            node, attr = node_attr
            curves[curve] = (node, attr)

    return curves


def getCurveTarget(curve):
    """Find the node and attribute that an animation curve controls."""
    if not cmds.objExists(curve):
        return None

    node_type = cmds.nodeType(curve)
    if not node_type.startswith('animCurve'):
        return None

    outputs = cmds.listConnections(f"{curve}.output", source=False,
                                    destination=True, plugs=True) or []

    for output_plug in outputs:
        if '.input' in output_plug.lower():
            target = traceBlendToTarget(output_plug)
            if target:
                return target
        elif '.' in output_plug:
            parts = output_plug.rsplit('.', 1)
            if len(parts) == 2:
                return (parts[0], parts[1])

    return None


def traceBlendToTarget(blend_plug):
    """Trace from a blend node input to the final target attribute."""
    if '.' not in blend_plug:
        return None

    blend_node, input_attr = blend_plug.split('.', 1)

    output_attr = "output"
    if input_attr.endswith('X'):
        output_attr = "outputX"
    elif input_attr.endswith('Y'):
        output_attr = "outputY"
    elif input_attr.endswith('Z'):
        output_attr = "outputZ"

    visited = set()
    current = blend_node
    current_output = output_attr

    while current and current not in visited:
        visited.add(current)

        full_output = f"{current}.{current_output}"
        outputs = []

        if cmds.objExists(full_output):
            outputs = cmds.listConnections(full_output, source=False,
                                            destination=True, plugs=True) or []

        if not outputs and current_output != "output":
            generic_output = f"{current}.output"
            if cmds.objExists(generic_output):
                outputs = cmds.listConnections(generic_output, source=False,
                                                destination=True, plugs=True) or []

        if not outputs:
            break

        for output_plug in outputs:
            if '.' not in output_plug:
                continue

            target_node, target_attr = output_plug.split('.', 1)
            target_type = cmds.nodeType(target_node)

            if 'Blend' in target_type or 'blend' in target_type or target_type == 'pairBlend':
                current = target_node
                if target_type == 'pairBlend':
                    if 'Translate' in target_attr:
                        axis = target_attr[-2] if target_attr[-1].isdigit() else target_attr[-1]
                        current_output = f"outTranslate{axis.upper()}"
                    elif 'Rotate' in target_attr:
                        axis = target_attr[-2] if target_attr[-1].isdigit() else target_attr[-1]
                        current_output = f"outRotate{axis.upper()}"
                break
            else:
                parts = output_plug.rsplit('.', 1)
                if len(parts) == 2:
                    return (parts[0], parts[1])
        else:
            break

    return None


def unrollAdditiveRotations(layer, nodes=None, debug=False):
    """
    Unroll rotation curves on an additive animation layer.

    This function accounts for the layer stack (composite base values),
    the layer's rotation accumulation mode, and animated layer weights.

    For additive layers, the "correct" Euler representation of delta values
    depends on context. This function:
    1. Computes the composed rotation result at each keyframe
    2. Euler filters the composed result to remove discontinuities
    3. Derives new delta values that produce the filtered composed result

    For component mode (rotationAccumulationMode=0):
        composed = base + (delta * weight)
        new_delta = (filtered_composed - base) / weight

    For quaternion mode (rotationAccumulationMode=1):
        composed = base_quat * slerp(identity, delta_quat, weight)
        new_delta = (inv(base) * filtered)^(1/weight)

    Args:
        layer: Name of the additive animation layer
        nodes: Optional list of nodes to process. If None, processes all nodes
               on the layer. Nodes must have all three rotation channels
               (rotateX, rotateY, rotateZ) on the layer to be processed.
        debug: If True, print diagnostic information about the filtering process.

    Returns:
        dict: {node: num_keys_processed} for each processed node

    Raises:
        ValueError: If layer doesn't exist or is in override mode
    """
    if not cmds.objExists(layer):
        raise ValueError(f"Layer '{layer}' does not exist")

    if cmds.nodeType(layer) != 'animLayer':
        raise ValueError(f"'{layer}' is not an animation layer")

    # Check if layer is additive (override = 0 means additive)
    is_override = cmds.getAttr(f"{layer}.override")
    if is_override:
        log.warning(f"Layer '{layer}' is in override mode. Use eulerFilterLayer() instead.")
        return {}

    # Get all nodes on the layer
    layer_attrs = cmds.animLayer(layer, query=True, attribute=True) or []
    layer_nodes = set(a.split('.')[0] for a in layer_attrs if '.' in a)

    if not layer_nodes:
        log.warning(f"No nodes found on layer '{layer}'")
        return {}

    # Filter to requested nodes if provided
    if nodes is not None:
        if isinstance(nodes, str):
            nodes = [nodes]
        # Only process nodes that are both requested AND on the layer
        nodes_to_process = [n for n in nodes if n in layer_nodes]
        if not nodes_to_process:
            log.warning(f"None of the specified nodes are on layer '{layer}'")
            return {}
    else:
        nodes_to_process = list(layer_nodes)

    processed = {}

    for node in nodes_to_process:
        try:
            # Build layer stack for this node
            stack = LayerStack.build(node, layer)

            # Check if we have all rotation curves - euler filter requires all 3 axes
            rot_attrs = ['rotateX', 'rotateY', 'rotateZ']
            missing_attrs = [attr for attr in rot_attrs if attr not in stack.target_curves]
            if missing_attrs:
                log.warning(
                    f"Skipping '{node}': missing rotation channels {missing_attrs} on layer. "
                    f"Euler filtering requires all 3 rotation axes."
                )
                continue

            curve_fns = {attr: stack.target_curves[attr] for attr in rot_attrs}

            # Get curve names for writing back
            curve_names = {
                attr: api.MFnDependencyNode(fn.object()).name()
                for attr, fn in curve_fns.items()
            }

            # Collect all key times (union of all rotation curves)
            key_times_set = set()
            for attr, curve_fn in curve_fns.items():
                for i in range(curve_fn.numKeys):
                    mtime = curve_fn.input(i)
                    key_times_set.add(mtime.asUnits(api.MTime.uiUnit()))

            key_times = sorted(key_times_set)

            if len(key_times) < 2:
                continue

            # Get rotation accumulation mode
            layer_info = stack.layers[layer]
            rotation_mode = layer_info.rotation_mode  # 0=component, 1=quaternion

            # Sample values at each key time
            delta_rx, delta_ry, delta_rz = [], [], []
            base_rx, base_ry, base_rz = [], [], []
            weights = []

            for t in key_times:
                mtime = api.MTime(t, api.MTime.uiUnit())

                # Get delta values from curves (convert from radians to degrees)
                delta_rx.append(math.degrees(curve_fns['rotateX'].evaluate(mtime)))
                delta_ry.append(math.degrees(curve_fns['rotateY'].evaluate(mtime)))
                delta_rz.append(math.degrees(curve_fns['rotateZ'].evaluate(mtime)))

                # Get base composite values
                base_rx.append(stack.get_base_value('rotateX', t))
                base_ry.append(stack.get_base_value('rotateY', t))
                base_rz.append(stack.get_base_value('rotateZ', t))

                # Get layer weight at this time
                weights.append(stack.get_layer_weight(layer, t))

            # Compute composed rotations
            composed_rx, composed_ry, composed_rz = [], [], []

            for i in range(len(key_times)):
                weight = weights[i]

                if rotation_mode == 0:
                    # Component mode: composed = base + (delta * weight)
                    composed_rx.append(base_rx[i] + delta_rx[i] * weight)
                    composed_ry.append(base_ry[i] + delta_ry[i] * weight)
                    composed_rz.append(base_rz[i] + delta_rz[i] * weight)
                else:
                    # Quaternion mode: composed = base_quat * slerp(identity, delta_quat, weight)
                    base_quat = eulerToQuaternion(
                        math.radians(base_rx[i]),
                        math.radians(base_ry[i]),
                        math.radians(base_rz[i])
                    )
                    delta_quat = eulerToQuaternion(
                        math.radians(delta_rx[i]),
                        math.radians(delta_ry[i]),
                        math.radians(delta_rz[i])
                    )

                    # slerp(identity, delta, weight) = delta^weight
                    weighted_delta = quaternionPower(delta_quat, weight)

                    # composed = base * weighted_delta
                    composed_quat = quaternionMultiply(base_quat, weighted_delta)

                    # Convert back to Euler
                    c_rx, c_ry, c_rz = quaternionToEuler(composed_quat)
                    composed_rx.append(math.degrees(c_rx))
                    composed_ry.append(math.degrees(c_ry))
                    composed_rz.append(math.degrees(c_rz))

            # Euler filter the composed rotations
            filtered_rx, filtered_ry, filtered_rz = eulerFilter(
                composed_rx, composed_ry, composed_rz
            )

            if debug:
                print(f"\n=== Debug for {node} ===")
                print(f"Rotation mode: {'component' if rotation_mode == 0 else 'quaternion'}")
                print(f"Num keys: {len(key_times)}")

                # Find indices where filtering made a difference
                changes = []
                for i in range(len(key_times)):
                    diff_rx = abs(filtered_rx[i] - composed_rx[i])
                    diff_ry = abs(filtered_ry[i] - composed_ry[i])
                    diff_rz = abs(filtered_rz[i] - composed_rz[i])
                    if diff_rx > 0.01 or diff_ry > 0.01 or diff_rz > 0.01:
                        changes.append((i, key_times[i], diff_rx, diff_ry, diff_rz))

                if changes:
                    print(f"Euler filter made {len(changes)} changes:")
                    for idx, t, dx, dy, dz in changes[:5]:  # Show first 5
                        print(f"  Frame {t}: composed=({composed_rx[idx]:.2f}, {composed_ry[idx]:.2f}, {composed_rz[idx]:.2f}) -> filtered=({filtered_rx[idx]:.2f}, {filtered_ry[idx]:.2f}, {filtered_rz[idx]:.2f})")
                else:
                    print("Euler filter made NO changes to composed rotations!")
                    # Show sample of composed values to see if there are discontinuities
                    print("Sample composed values (looking for ~360 degree jumps):")
                    for i in range(min(10, len(key_times))):
                        print(f"  Frame {key_times[i]}: composed=({composed_rx[i]:.2f}, {composed_ry[i]:.2f}, {composed_rz[i]:.2f})")
                    if len(key_times) > 10:
                        print("  ...")

                print(f"\nSample base values:")
                for i in range(min(5, len(key_times))):
                    print(f"  Frame {key_times[i]}: base=({base_rx[i]:.2f}, {base_ry[i]:.2f}, {base_rz[i]:.2f})")

                print(f"\nSample delta values (from layer curves):")
                for i in range(min(5, len(key_times))):
                    print(f"  Frame {key_times[i]}: delta=({delta_rx[i]:.2f}, {delta_ry[i]:.2f}, {delta_rz[i]:.2f})")

            # Derive new deltas
            new_delta_rx, new_delta_ry, new_delta_rz = [], [], []

            for i in range(len(key_times)):
                weight = weights[i]

                if abs(weight) < 1e-10:
                    # Weight is zero, delta has no effect - keep original
                    new_delta_rx.append(delta_rx[i])
                    new_delta_ry.append(delta_ry[i])
                    new_delta_rz.append(delta_rz[i])
                elif rotation_mode == 0:
                    # Component mode: new_delta = (filtered_composed - base) / weight
                    new_delta_rx.append((filtered_rx[i] - base_rx[i]) / weight)
                    new_delta_ry.append((filtered_ry[i] - base_ry[i]) / weight)
                    new_delta_rz.append((filtered_rz[i] - base_rz[i]) / weight)
                else:
                    # Quaternion mode: new_delta = (inv(base) * filtered)^(1/weight)
                    base_quat = eulerToQuaternion(
                        math.radians(base_rx[i]),
                        math.radians(base_ry[i]),
                        math.radians(base_rz[i])
                    )
                    filtered_quat = eulerToQuaternion(
                        math.radians(filtered_rx[i]),
                        math.radians(filtered_ry[i]),
                        math.radians(filtered_rz[i])
                    )

                    # S = inv(base) * filtered (what the weighted delta should be)
                    base_inv = quaternionInverse(base_quat)
                    weighted_delta_needed = quaternionMultiply(base_inv, filtered_quat)

                    # new_delta = S^(1/weight) to un-weight it
                    new_delta_quat = quaternionPower(weighted_delta_needed, 1.0 / weight)

                    # Convert to Euler
                    nd_rx, nd_ry, nd_rz = quaternionToEuler(new_delta_quat)
                    new_delta_rx.append(math.degrees(nd_rx))
                    new_delta_ry.append(math.degrees(nd_ry))
                    new_delta_rz.append(math.degrees(nd_rz))

            # For quaternion mode, the quaternion-to-euler conversion gives "canonical"
            # values that may undo our unwrapping. Apply euler filter to the new deltas
            # to ensure they're continuous (the deltas are what get stored in curves).
            if rotation_mode == 1:
                new_delta_rx, new_delta_ry, new_delta_rz = eulerFilter(
                    new_delta_rx, new_delta_ry, new_delta_rz
                )

            if debug:
                # Show changes in delta values
                delta_changes = []
                for i in range(len(key_times)):
                    diff_rx = abs(new_delta_rx[i] - delta_rx[i])
                    diff_ry = abs(new_delta_ry[i] - delta_ry[i])
                    diff_rz = abs(new_delta_rz[i] - delta_rz[i])
                    if diff_rx > 0.01 or diff_ry > 0.01 or diff_rz > 0.01:
                        delta_changes.append((i, key_times[i]))

                print(f"\nDelta values changed at {len(delta_changes)} frames")
                if delta_changes:
                    for idx, t in delta_changes[:5]:
                        print(f"  Frame {t}: old=({delta_rx[idx]:.2f}, {delta_ry[idx]:.2f}, {delta_rz[idx]:.2f}) -> new=({new_delta_rx[idx]:.2f}, {new_delta_ry[idx]:.2f}, {new_delta_rz[idx]:.2f})")

            # Build time->value lookup for writing back
            # Round times to avoid float precision issues in dictionary lookup
            def round_time(t):
                return round(t, 6)

            time_to_value = {
                'rotateX': {round_time(t): v for t, v in zip(key_times, new_delta_rx)},
                'rotateY': {round_time(t): v for t, v in zip(key_times, new_delta_ry)},
                'rotateZ': {round_time(t): v for t, v in zip(key_times, new_delta_rz)},
            }

            # Write new deltas back to curves
            for attr, curve_fn in curve_fns.items():
                curve_name = curve_names[attr]
                value_lookup = time_to_value[attr]

                for i in range(curve_fn.numKeys):
                    mtime = curve_fn.input(i)
                    t = round_time(mtime.asUnits(api.MTime.uiUnit()))

                    if t in value_lookup:
                        cmds.keyframe(curve_name, index=(i,),
                                      valueChange=value_lookup[t], absolute=True)

            processed[node] = len(key_times)

        except Exception as e:
            log.error(f"Error unrolling rotations for '{node}': {e}")

    return processed


# =============================================================================
# Transform Recording (for debugging)
# =============================================================================

@viewport_off
def record_transforms(controls, start_frame, end_frame, sample=1):
    """Record world-space transforms for controls across frame range."""
    transforms = {}
    current_time = cmds.currentTime(query=True)

    for ctrl in controls:
        transforms[ctrl] = {}
        for frame in range(int(start_frame), int(end_frame) + 1, sample):
            cmds.currentTime(frame, edit=True)
            ws_pos = cmds.xform(ctrl, query=True, worldSpace=True, translation=True)
            ws_rot = cmds.xform(ctrl, query=True, worldSpace=True, rotation=True)
            transforms[ctrl][frame] = (ws_pos, ws_rot)

    cmds.currentTime(current_time, edit=True)
    return transforms


def get_xform_at_frame(ctrl, frame):
    """Get world-space translation and rotation at a specific frame."""
    cmds.currentTime(frame, edit=True)
    ws_pos = cmds.xform(ctrl, query=True, worldSpace=True, translation=True)
    ws_rot = cmds.xform(ctrl, query=True, worldSpace=True, rotation=True)
    return ws_pos, ws_rot

# =============================================================================
# UI Constants
# =============================================================================

WIDTH = 180
HEIGHT = 462

# =============================================================================
# Helper Functions
# =============================================================================

def _get_maya_window():
    """Get the main Maya window as a Qt widget."""
    ptr = mui.MQtUtil.mainWindow()
    return QtCompat.wrapInstance(int(ptr), QtWidgets.QMainWindow)


# =============================================================================
# Public API Functions
# =============================================================================

@undo
def create_pins(selection=None, start_frame=None, end_frame=None, group_override=None):
    """
    Create world-space pins for the selected controls.

    Args:
        selection: Controls to pin (default: current selection)
        start_frame: Start frame for pin range (default: playback range)
        end_frame: End frame for pin range (default: playback range)
        group_override: Optional custom master group name

    Returns:
        str: Name of the created pin group, or None on failure
    """
    # Validate animation blending preference
    current_state = None
    if cmds.optionVar(exists='animBlendingOpt'):
        current_state = cmds.optionVar(query='animBlendingOpt')
    if not current_state:
        cmds.error('Animation blending preference is NOT SET!')
        return None

    sel_list = get_selectionList(selection)
    controls = validate_selection(sel_list)
    if controls.isEmpty():
        api.MGlobal.displayError("Could not find any valid nodes to pin.")
        return None

    start_frame, end_frame = validate_framerange(start_frame, end_frame)
    if start_frame is None or end_frame is None:
        api.MGlobal.displayError(
            f"Could not validate frame range from {start_frame} to {end_frame}.")
        return None

    new_pin_group = create_new_pin_group()
    if not new_pin_group:
        api.MGlobal.displayError("Could not create a valid group to add pins.")
        return None

    # Collect control names for bookending (use long names for unique identification)
    control_names = [get_long_name(controls.getDependNode(i))
                     for i in range(controls.length())]

    # Bookend curves to preserve shape outside work area
    bookend_curves(control_names, start_frame, end_frame)

    # Create pins and constraints
    locators = []
    constraints = []
    for i in range(controls.length()):
        control = controls.getDependNode(i)
        # Use long name for unique identification when there are duplicate short names
        control_name = get_long_name(control)
        control_data = read_control_data(control, start_frame, end_frame)
        locator = create_locator_pin(control_data, new_pin_group)
        locators.append(locator)
        constraint_node = cmds.parentConstraint(control_name, locator)[0]
        constraints.append(constraint_node)
        cmds.setKeyframe(locator)
        cmds.setAttr(locator + '.blendParent1', 1)

    # Bake the locators
    results = do_bake(locators, start_frame, end_frame)
    if not results:
        raise ValueError("Bake failed.")
    cmds.delete(constraints)

    # Set up reverse constraints (pins drive controls)
    pins = get_pins(new_pin_group)
    for pin in pins:
        control = cmds.getAttr(pin + '.control')
        MSel = api.MGlobal.getSelectionListByName(control)
        controlFN = api.MFnDependencyNode(MSel.getDependNode(0))

        if is_locked_or_not_keyable(controlFN, 'translate'):
            skip_translate = ['x', 'y', 'z']
        else:
            skip_translate = 'none'
        if is_locked_or_not_keyable(controlFN, 'rotate'):
            skip_rotate = ['x', 'y', 'z']
        else:
            skip_rotate = 'none'

        constraint_node = cmds.parentConstraint(
            pin, control,
            skipTranslate=skip_translate,
            skipRotate=skip_rotate)[0]
        cmds.setAttr(pin + '.constraint', constraint_node, type='string')

        try:
            cmds.setKeyframe(control, attribute="blendParent1",
                             time=[start_frame, end_frame], value=1)
            cmds.setKeyframe(control, attribute="blendParent1",
                             time=[start_frame - 1, end_frame + 1], value=0)
        except:
            api.MGlobal.displayWarning(
                "Could not key the blendParent. Be careful outside the buffer range!")

        # Lock pin attributes that were locked on control
        t_lock = cmds.getAttr(pin + '.translate_locked')
        r_lock = cmds.getAttr(pin + '.rotate_locked')
        cmds.setAttr(locator + '.t', lock=t_lock, keyable=(not t_lock))
        cmds.setAttr(locator + '.r', lock=r_lock, keyable=(not r_lock))

    # Select newly created pins
    new_pins = get_pins([new_pin_group])
    if new_pins:
        cmds.select(new_pins, replace=True)

    return new_pin_group


@undo
def create_relative_pins(selection=None, start_frame=None, end_frame=None):
    """
    Create pins that hold controls relative to a parent control's space.

    Uses Maya constraint paradigm: selection[0] is the parent space,
    selection[1:] are the controls to pin into that space.

    Args:
        selection: List of controls (first is parent, rest are children)
        start_frame: Start frame for bake
        end_frame: End frame for bake

    Returns:
        str: Name of the pin group created, or None on failure
    """
    # Validate animation blending preference
    current_state = None
    if cmds.optionVar(exists='animBlendingOpt'):
        current_state = cmds.optionVar(query='animBlendingOpt')
    if not current_state:
        cmds.error('Animation blending preference is NOT SET!')
        return None

    sel_list = get_selectionList(selection)
    if sel_list.length() < 2:
        api.MGlobal.displayError(
            "Select a parent control first, then the controls to pin into that space.")
        return None

    # First selection is the parent space
    parent_node = sel_list.getDependNode(0)
    parent_name = get_long_name(parent_node)
    parent_short = get_short_name(parent_name)

    # Build list of child control names (use long names for unique identification)
    child_names = []
    for i in range(1, sel_list.length()):
        node = sel_list.getDependNode(i)
        child_names.append(get_long_name(node))

    # Validate children
    child_sel = api.MSelectionList()
    for name in child_names:
        child_sel.add(name)
    controls = validate_selection(child_sel)
    if controls.isEmpty():
        api.MGlobal.displayError("Could not find any valid child nodes to pin.")
        return None

    start_frame, end_frame = validate_framerange(start_frame, end_frame)
    if start_frame is None or end_frame is None:
        api.MGlobal.displayError(
            f"Could not validate frame range from {start_frame} to {end_frame}.")
        return None

    # Collect control names for bookending (use long names for unique identification)
    control_names = [get_long_name(controls.getDependNode(i))
                     for i in range(controls.length())]
    bookend_curves(control_names, start_frame, end_frame)

    # Create pin_group constrained to parent
    new_pin_group = create_new_pin_group()
    if not new_pin_group:
        api.MGlobal.displayError("Could not create a valid group to add pins.")
        return None

    space_constraint = cmds.parentConstraint(parent_name, new_pin_group, maintainOffset=False)[0]
    cmds.parent(space_constraint, new_pin_group)

    # Lock all channels on pin_group
    for attr in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']:
        cmds.setAttr(f'{new_pin_group}.{attr}', lock=True)

    # Create pins for each child control
    locators = []
    temp_constraints = []

    for i in range(controls.length()):
        control = controls.getDependNode(i)
        # Use long name for unique identification when there are duplicate short names
        control_name = get_long_name(control)
        control_data = read_control_data(control, start_frame, end_frame)

        locator = create_locator_pin(control_data, new_pin_group)
        locators.append(locator)

        rotate_order = cmds.getAttr(f'{control_name}.rotateOrder')
        cmds.setAttr(f'{locator}.rotateOrder', rotate_order)

        constraint_node = cmds.parentConstraint(control_name, locator, maintainOffset=False)[0]
        temp_constraints.append(constraint_node)

    # Bake locators
    results = do_bake(locators, start_frame, end_frame)
    if not results:
        raise ValueError("Bake failed.")
    cmds.delete(temp_constraints)

    # Reverse setup - constrain controls to pins
    for locator in locators:
        control = cmds.getAttr(locator + '.control')
        MSel = api.MGlobal.getSelectionListByName(control)
        controlFN = api.MFnDependencyNode(MSel.getDependNode(0))

        if is_locked_or_not_keyable(controlFN, 'translate'):
            skip_translate = ['x', 'y', 'z']
        else:
            skip_translate = 'none'
        if is_locked_or_not_keyable(controlFN, 'rotate'):
            skip_rotate = ['x', 'y', 'z']
        else:
            skip_rotate = 'none'

        constraint_node = cmds.parentConstraint(
            locator, control,
            maintainOffset=False,
            skipTranslate=skip_translate,
            skipRotate=skip_rotate)[0]

        constraint_grp = cmds.group(empty=True, name=locator + '_constraints')
        cmds.parent(constraint_grp, locator)
        cmds.parent(constraint_node, constraint_grp)

        cmds.setAttr(locator + '.constraint', constraint_node, type='string')

        try:
            cmds.setKeyframe(control, attribute="blendParent1",
                             time=[start_frame, end_frame], value=1)
            cmds.setKeyframe(control, attribute="blendParent1",
                             time=[start_frame - 1, end_frame + 1], value=0)
        except:
            api.MGlobal.displayWarning(
                "Could not key the blendParent. Be careful outside the buffer range!")

        t_lock = cmds.getAttr(locator + '.translate_locked')
        r_lock = cmds.getAttr(locator + '.rotate_locked')
        cmds.setAttr(locator + '.t', lock=t_lock, keyable=(not t_lock))
        cmds.setAttr(locator + '.r', lock=r_lock, keyable=(not r_lock))

    # Select newly created pins
    new_pins = get_pins([new_pin_group])
    if new_pins:
        cmds.select(new_pins, replace=True)

    api.MGlobal.displayInfo(
        f"Created relative pins for {controls.length()} control(s) in '{parent_short}' space.")
    return new_pin_group


@undo
def bake_pins(pin_groups=None, bake_option=1, start_frame=None, end_frame=None,
              animLayer=False, unrollRotations=True):
    """
    Bake pinned controls back to their original animation.

    Args:
        pin_groups: Pin groups to bake (default: selected pin groups)
        bake_option: 0 = Match Keys, >0 = Bake step interval
        start_frame: Start frame for bake
        end_frame: End frame for bake
        animLayer: If True, bake to additive animation layer using Hybrid approach
        unrollRotations: If True (and animLayer=True), apply euler filter

    Returns:
        Set of baked pin groups
    """
    if not pin_groups:
        selection = cmds.ls(sl=True)
        pin_group_list = find_pin_groups_from_selection(selection)
    else:
        if isinstance(pin_groups, str):
            pin_group_list = {pin_groups}
        else:
            pin_group_list = set(pin_groups)

    pins_to_bake = get_pins(pin_group_list)

    if not pins_to_bake:
        api.MGlobal.displayWarning("No pins could be found to bake.")
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
            # Fallback: recover control from constraint connections
            control_list = cmds.listConnections(
                pin_constraint + '.constraintParentInverseMatrix') or []
            if control_list:
                # Use long name for unique identification
                control = cmds.ls(control_list[0], long=True)[0]
                cmds.setAttr(pin + '.control', control, type='string')
        controls_to_bake.append(control)
        pin_parent = cmds.listRelatives(pin, parent=True)[0]
        pin_groups_to_delete.add(pin_parent)
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        blendParents_to_restore[control] = bp_keys

    start_frame, end_frame = validate_bakerange(pins_to_bake, start_frame, end_frame)
    if start_frame is None or end_frame is None:
        api.MGlobal.displayError(
            f"Could not validate frame range from {start_frame} to {end_frame}.")
        return None

    sample = 1 if bake_option == 0 else bake_option

    # Bake
    if animLayer:
        success = do_bake_to_layer(
            controls_to_bake, start_frame, end_frame, sample,
            unrollRotations=unrollRotations, constraints=constraints)
    else:
        success = do_bake(controls_to_bake, start_frame, end_frame, sample)

    if not success:
        raise ValueError("Bake failed.")

    # Clean up blendParent attributes
    for control, bp_keys in blendParents_to_restore.items():
        bp_attr = f'{control}.blendParent1'
        if not cmds.objExists(bp_attr):
            continue

        cmds.cutKey(control, attribute='blendParent1', clear=True)

        for match in tuple_rex.finditer(bp_keys):
            cmds.setKeyframe(
                control, at='blendParent1',
                time=float(match.group(0).split(', ')[0]),
                value=float(match.group(0).split(', ')[1]))

        if not tuple_rex.search(bp_keys):
            try:
                cmds.deleteAttr(bp_attr)
            except:
                pass

    if bake_option == 0:
        success = match_keys_procedure(pins_to_bake, start_frame, end_frame)
        if not success:
            raise ValueError("match_keys_procedure failed.")

    # Cleanup
    remaining_constraints = [c for c in constraints if cmds.objExists(c)]
    if remaining_constraints:
        cmds.delete(remaining_constraints)
    cmds.delete(list(pin_groups_to_delete))

    # Delete master group if empty
    global MASTER_GROUP
    pins_exist = get_pins()
    if not pins_exist and cmds.objExists(MASTER_GROUP):
        cmds.delete(MASTER_GROUP)

    for pin_group in list(pin_groups_to_delete):
        print(pin_group)
    return pin_groups_to_delete


def select_all_pins():
    """
    Select all pins in the scene.

    Returns:
        list: Names of all selected pins, or empty list if none found
    """
    all_pins = get_pins()
    if all_pins:
        cmds.select(all_pins, replace=True)
    else:
        cmds.select(clear=True)
        api.MGlobal.displayWarning("No pins found in the scene.")
    return all_pins


@undo
def smart_bake_pins(pin_groups=None, bake_option=1, start_frame=None, end_frame=None,
                    unrollRotations=True):
    """
    Smart bake that auto-detects whether to use animation layers per-control.

    Each control is checked individually - controls with animation on layers
    are baked to an additive layer, while controls without layers are baked
    destructively. This correctly handles mixed scenarios where some controls
    have layers and others don't.

    Args:
        pin_groups: Pin groups to bake (default: selected pin groups)
        bake_option: 0 = Match Keys, >0 = bake step interval
        start_frame: Start frame for bake
        end_frame: End frame for bake
        unrollRotations: If True, apply euler filter when baking to layer

    Returns:
        Set of baked pin groups
    """
    if not pin_groups:
        selection = cmds.ls(sl=True)
        pin_group_list = find_pin_groups_from_selection(selection)
    else:
        if isinstance(pin_groups, str):
            pin_group_list = {pin_groups}
        else:
            pin_group_list = set(pin_groups)

    all_pins = get_pins(pin_group_list)

    if not all_pins:
        api.MGlobal.displayWarning("No pins could be found to bake.")
        return None

    # Separate pins by whether their control has animation layers
    layer_pins = []
    non_layer_pins = []
    for pin in all_pins:
        control = cmds.getAttr(pin + '.control')
        if control_has_anim_layers(control):
            layer_pins.append(pin)
        else:
            non_layer_pins.append(pin)

    # Report what we found
    if layer_pins and non_layer_pins:
        api.MGlobal.displayInfo(
            f"Mixed mode: {len(layer_pins)} control(s) with layers, "
            f"{len(non_layer_pins)} without - baking each appropriately")
    elif layer_pins:
        api.MGlobal.displayInfo("All controls have animation layers - baking to additive layer")
    else:
        api.MGlobal.displayInfo("No animation layers detected - baking destructively")

    # Bake each group with appropriate settings
    result_groups = set()

    if non_layer_pins:
        result = _bake_specific_pins(
            non_layer_pins, pin_group_list, bake_option,
            start_frame, end_frame, animLayer=False, unrollRotations=False)
        if result:
            result_groups.update(result)

    if layer_pins:
        result = _bake_specific_pins(
            layer_pins, pin_group_list, bake_option,
            start_frame, end_frame, animLayer=True, unrollRotations=unrollRotations)
        if result:
            result_groups.update(result)

    return result_groups


def _bake_specific_pins(pins_to_bake, all_pin_groups, bake_option=1,
                        start_frame=None, end_frame=None,
                        animLayer=False, unrollRotations=True):
    """
    Internal helper to bake a specific list of pins.

    This is used by smart_bake_pins to bake subsets of pins with different settings.
    """
    if not pins_to_bake:
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
            # Fallback: recover control from constraint connections
            control_list = cmds.listConnections(
                pin_constraint + '.constraintParentInverseMatrix') or []
            if control_list:
                control = cmds.ls(control_list[0], long=True)[0]
                cmds.setAttr(pin + '.control', control, type='string')
        controls_to_bake.append(control)
        pin_parent = cmds.listRelatives(pin, parent=True)[0]
        pin_groups_to_delete.add(pin_parent)
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        blendParents_to_restore[control] = bp_keys

    start_frame, end_frame = validate_bakerange(pins_to_bake, start_frame, end_frame)
    if start_frame is None or end_frame is None:
        api.MGlobal.displayError(
            f"Could not validate frame range from {start_frame} to {end_frame}.")
        return None

    sample = 1 if bake_option == 0 else bake_option

    # Bake
    if animLayer:
        success = do_bake_to_layer(
            controls_to_bake, start_frame, end_frame, sample,
            unrollRotations=unrollRotations, constraints=constraints)
    else:
        success = do_bake(controls_to_bake, start_frame, end_frame, sample)

    if not success:
        raise ValueError("Bake failed.")

    # Clean up blendParent attributes
    for control, bp_keys in blendParents_to_restore.items():
        bp_attr = f'{control}.blendParent1'
        if not cmds.objExists(bp_attr):
            continue

        cmds.cutKey(control, attribute='blendParent1', clear=True)

        for match in tuple_rex.finditer(bp_keys):
            cmds.setKeyframe(
                control, at='blendParent1',
                time=float(match.group(0).split(', ')[0]),
                value=float(match.group(0).split(', ')[1]))

        if not tuple_rex.search(bp_keys):
            try:
                cmds.deleteAttr(bp_attr)
            except:
                pass

    if bake_option == 0:
        success = match_keys_procedure(pins_to_bake, start_frame, end_frame)
        if not success:
            raise ValueError("match_keys_procedure failed.")

    # Cleanup constraints
    remaining_constraints = [c for c in constraints if cmds.objExists(c)]
    if remaining_constraints:
        cmds.delete(remaining_constraints)

    # Delete pins (but not pin groups yet - they might have other pins)
    for pin in pins_to_bake:
        if cmds.objExists(pin):
            cmds.delete(pin)

    # Clean up empty pin groups
    for pg in list(pin_groups_to_delete):
        if cmds.objExists(pg):
            children = cmds.listRelatives(pg, children=True) or []
            if not children:
                cmds.delete(pg)

    # Delete master group if empty
    global MASTER_GROUP
    pins_exist = get_pins()
    if not pins_exist and cmds.objExists(MASTER_GROUP):
        cmds.delete(MASTER_GROUP)

    return pin_groups_to_delete


# =============================================================================
# UI Class
# =============================================================================

class AnimPinUI(QtWidgets.QDialog):
    """Animation Pin Tool UI"""

    UI_INSTANCE = None

    # OptionVar names for persistent preferences
    PREF_BAKE_OPTION = "animPinTool_bakeOption"
    PREF_BAKE_STEP = "animPinTool_bakeStep"
    PREF_BAKE_TO_LAYER = "animPinTool_bakeToLayer"
    PREF_UNROLL_ROTATIONS = "animPinTool_unrollRotations"

    @classmethod
    def _is_valid_widget(cls, widget):
        """Check if a Qt widget is still valid."""
        if widget is None:
            return False
        return QtCompat.isValid(widget)

    @classmethod
    def run(cls):
        """Show the UI. Creates a new instance if needed."""
        print("animPinUI version: {}".format(__version__))
        if cls.UI_INSTANCE is not None and not cls._is_valid_widget(cls.UI_INSTANCE):
            cls.UI_INSTANCE = None

        if cls.UI_INSTANCE is None:
            cls.UI_INSTANCE = AnimPinUI()

        if cls.UI_INSTANCE.isHidden():
            cls.UI_INSTANCE.show()
        else:
            cls.UI_INSTANCE.raise_()
            cls.UI_INSTANCE.activateWindow()

    @classmethod
    def stop(cls):
        """Close and destroy the UI."""
        if cls.UI_INSTANCE is not None and cls._is_valid_widget(cls.UI_INSTANCE):
            cls.UI_INSTANCE.close()
        cls.UI_INSTANCE = None

    def __init__(self, parent=None):
        super(AnimPinUI, self).__init__(parent)
        self.parent = _get_maya_window()
        self.setParent(self.parent)
        self.setWindowFlags(
            QtCore.Qt.Dialog |
            QtCore.Qt.WindowCloseButtonHint)
        self.setObjectName('AnimPin')
        self.setWindowTitle('Animation Pin Tool')
        self.setProperty("saveWindowPref", True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # Class globals
        self.pressPos = None
        self.isMoving = False
        self._callbacks = {}
        self.width = WIDTH
        self.height = HEIGHT
        self.mini_state = False

        # Build UI
        self.build_UI()
        self._load_preferences()
        self.init_connections()
        self.init_frame_range()
        self._init_pin_group_list()

    def init_connections(self):
        """Set up signal/slot connections."""
        self.BTN_select_space.clicked.connect(self.on_select_space)
        self.BTN_create_pins.clicked.connect(self.on_create_pins)
        self.BTN_create_relative_pins.clicked.connect(self.on_create_relative_pins)
        self.BTN_bake_pins.clicked.connect(self.on_bake_pins)
        self.destroyed.connect(self.closeEvent)

        self.LST_pin_groups.itemSelectionChanged.connect(self._pass_selection_to_maya)
        self.LST_pin_groups.itemChanged.connect(self._ui_pin_name_changed)

        self.OPT_0_matchkeys.toggled.connect(self._save_preferences)
        self.SPN_step.valueChanged.connect(self._save_preferences)
        self.CHK_bake_to_layer.toggled.connect(self._save_preferences)
        self.CHK_unroll_rotations.toggled.connect(self._save_preferences)

    def init_frame_range(self):
        """Initialize frame range spinboxes from timeline."""
        spin_start = cmds.playbackOptions(query=True, animationStartTime=True)
        spin_end = cmds.playbackOptions(query=True, animationEndTime=True)
        self.SPN_start_frame.setValue(spin_start)
        self.SPN_end_frame.setValue(spin_end)

    def _load_preferences(self):
        """Load saved UI preferences from Maya optionVars."""
        if cmds.optionVar(exists=self.PREF_BAKE_OPTION):
            bake_option = cmds.optionVar(q=self.PREF_BAKE_OPTION)
            if bake_option == 0:
                self.OPT_0_matchkeys.setChecked(True)
            else:
                self.OPT_1_bakeStep.setChecked(True)

        if cmds.optionVar(exists=self.PREF_BAKE_STEP):
            step = cmds.optionVar(q=self.PREF_BAKE_STEP)
            self.SPN_step.setValue(step)

        if cmds.optionVar(exists=self.PREF_BAKE_TO_LAYER):
            checked = cmds.optionVar(q=self.PREF_BAKE_TO_LAYER)
            self.CHK_bake_to_layer.setChecked(bool(checked))

        if cmds.optionVar(exists=self.PREF_UNROLL_ROTATIONS):
            checked = cmds.optionVar(q=self.PREF_UNROLL_ROTATIONS)
            self.CHK_unroll_rotations.setChecked(bool(checked))

    def _save_preferences(self):
        """Save UI preferences to Maya optionVars."""
        bake_option = 0 if self.OPT_0_matchkeys.isChecked() else 1
        cmds.optionVar(iv=(self.PREF_BAKE_OPTION, bake_option))
        cmds.optionVar(iv=(self.PREF_BAKE_STEP, self.SPN_step.value()))
        cmds.optionVar(iv=(self.PREF_BAKE_TO_LAYER, int(self.CHK_bake_to_layer.isChecked())))
        cmds.optionVar(iv=(self.PREF_UNROLL_ROTATIONS, int(self.CHK_unroll_rotations.isChecked())))

    def build_UI(self):
        """Build the UI layout."""
        self._apply_stylesheet()

        # Main layout
        self.LYT_main_grid = QtWidgets.QGridLayout()
        self.setLayout(self.LYT_main_grid)
        self.LYT_main_grid.setContentsMargins(0, 0, 0, 0)
        self.LYT_main_grid.setSpacing(6)
        self.LYT_main_grid.setRowStretch(1, 1)  # Row 1 (main content) expands

        # Header image
        qpix = QtGui.QPixmap()
        qpix.loadFromData(IMAGE_DATA)
        self.LBL_header_image = QtWidgets.QLabel()
        self.LYT_main_grid.addWidget(self.LBL_header_image, 0, 0)
        self.LBL_header_image.setObjectName("headerLabel")
        self.LBL_header_image.setPixmap(qpix)
        self.LBL_header_image.setAlignment(QtCore.Qt.AlignCenter)
        self.LBL_header_image.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)

        # Main vertical layout
        self.LYT_main_vertical = QtWidgets.QVBoxLayout()
        self.LYT_main_grid.addLayout(self.LYT_main_vertical, 1, 0)
        self.LYT_main_vertical.setSpacing(8)
        self.LYT_main_vertical.setContentsMargins(10, 4, 10, 10)

        self._build_frame_range_section()
        self._add_line_divider()
        self._build_options_section()
        self._add_line_divider()
        self._build_buttons_section()
        self._add_line_divider()
        self._build_pin_list_section()
        self._build_context_menu()

    def _apply_stylesheet(self):
        """Apply the UI stylesheet."""
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
            QSpinBox {
                padding: 0px 8px 0px 5px;
                background-color: rgb(50, 50, 50);
                border-width: 0px;
                border-radius: 8px;
                color: rgb(150, 150, 150);
                font: bold 14pt Sans-serif;
            }
            QSpinBox:focus { background-color: rgb(55, 55, 55); }
            QSpinBox:hover { background-color: rgb(60, 60, 60); }
            QRadioButton {
                background-color: rgb(65, 65, 65);
                color: rgb(180, 180, 180);
                border-radius:8px;
                padding: 4px;
            }
            QRadioButton:checked { background-color: rgb(80, 80, 80); }
            QRadioButton:hover { background-color: rgb(90, 90, 90); }
            QRadioButton::indicator { width: 8px; height: 8px; border-radius: 6px; }
            QRadioButton::indicator:checked {
                background-color: #05B8CC;
                border: 2px solid grey;
                border-color: rgb(180, 180, 180);
            }
            QRadioButton::indicator:unchecked {
                background-color: rgb(60, 60, 60);
                border: 2px solid grey;
                border-color: rgb(140, 140, 140);
            }
            QPushButton {
                background-color: rgb(80, 80, 80);
                border-style: solid;
                border-width: 0px;
                border-radius: 8px;
                color: rgb(186, 186, 186);
                min-height: 50px;
            }
            QPushButton:hover { background-color: rgb(90, 90, 90); }
            QPushButton:pressed { background-color: rgb(74, 105, 129); }
            QListWidget {
                show-decoration-selected: 1;
                background: rgb(65, 65, 65);
                border: 1px solid grey;
                border-radius: 10px;
                padding: 6px;
                border-color: rgb(80, 80, 80);
            }
            QListWidget::item {
                background: rgb(65, 65, 65);
                margin-bottom: 2px;
                border-radius: 4px;
                padding-left: 4px;
                height: 24px;
            }
            QListWidget::item:selected { background-color: #4a6981; }
            QListWidget::item:hover { background: rgb(80, 80, 80); }
            QScrollBar:vertical {
                border: 1px solid grey;
                border-color: rgb(90,90,90);
                background: rgb(60,60,60);
                width: 8px;
            }
            QScrollBar::handle:vertical { background: rgb(90,90,90); min-height: 20px; }
            QMenu {
                background: rgb(65, 65, 65);
                border: 1px solid rgb(115, 115, 115);
                padding: 8px;
            }
            QMenu::item { color: rgb(180, 180, 180); padding: 4px 25px 4px 20px; }
            QMenu::item:selected { background: rgb(45, 45, 45); border-radius: 6px; }
            QCheckBox {
            background-color: rgb(65, 65, 65);
            color: rgb(180, 180, 180);
            border-radius: 8px;
            padding: 4px;
            }
            QCheckBox:checked { background-color: rgb(80, 80, 80); }
            QCheckBox:hover { background-color: rgb(90, 90, 90); }
            QCheckBox::indicator { width: 8px; height: 8px; border-radius: 0px; }
            QCheckBox::indicator:checked {
                background-color: #05B8CC;
                border: 2px solid grey;
                border-color: rgb(180, 180, 180);
            }
            QCheckBox::indicator:unchecked {
                background-color: rgb(60, 60, 60);
                border: 2px solid grey;
                border-color: rgb(140, 140, 140);
            }                           
        """)

    def _build_frame_range_section(self):
        """Build the start/end frame section."""
        self.LYT_grid_time = QtWidgets.QGridLayout()
        self.LYT_main_vertical.addLayout(self.LYT_grid_time)
        self.LYT_grid_time.setContentsMargins(0, 0, 0, 2)
        self.LYT_grid_time.setColumnStretch(1, 2)

        self.LBL_start_frame = QtWidgets.QLabel("Start Frame")
        self.LYT_grid_time.addWidget(self.LBL_start_frame, 0, 0)

        self.SPN_start_frame = QtWidgets.QSpinBox()
        self.LYT_grid_time.addWidget(self.SPN_start_frame, 0, 1)
        self.SPN_start_frame.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_start_frame.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_start_frame.setRange(-999999999, 999999999)

        self.LBL_end_frame = QtWidgets.QLabel("End Frame")
        self.LYT_grid_time.addWidget(self.LBL_end_frame, 1, 0)

        self.SPN_end_frame = QtWidgets.QSpinBox()
        self.LYT_grid_time.addWidget(self.SPN_end_frame, 1, 1)
        self.SPN_end_frame.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_end_frame.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_end_frame.setRange(-999999999, 999999999)

    def _build_options_section(self):
        """Build the bake options section."""
        self.GRP_options = QtWidgets.QGroupBox("Bake Options")
        self.LYT_main_vertical.addWidget(self.GRP_options)
        # Use Minimum policy - widget can grow but won't shrink below minimumHeight
        self.GRP_options.setMinimumHeight(140)
        self.GRP_options.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Minimum)

        self.LYT_options = QtWidgets.QVBoxLayout()
        self.GRP_options.setLayout(self.LYT_options)
        self.LYT_options.setSpacing(6)
        self.LYT_options.setContentsMargins(10, 12, 10, 4)

        self.OPT_0_matchkeys = QtWidgets.QRadioButton("Match Keys")
        self.LYT_options.addWidget(self.OPT_0_matchkeys)
        self.OPT_0_matchkeys.setChecked(True)

        self.LYT_step = QtWidgets.QHBoxLayout()
        self.LYT_options.addLayout(self.LYT_step)

        self.OPT_1_bakeStep = QtWidgets.QRadioButton("Bake step ")
        self.OPT_1_bakeStep.setMinimumWidth(self.OPT_1_bakeStep.sizeHint().width())
        self.OPT_1_bakeStep.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed)
        self.LYT_step.addWidget(self.OPT_1_bakeStep, 0)

        self.SPN_step = QtWidgets.QSpinBox()
        self.SPN_step.setMinimumWidth(40)
        self.LYT_step.addWidget(self.SPN_step, 1)
        self.SPN_step.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_step.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_step.setRange(1, 999999999)
        self.SPN_step.setValue(2)

        self.CHK_bake_to_layer = QtWidgets.QCheckBox("Bake to Animation Layer")
        self.LYT_options.addWidget(self.CHK_bake_to_layer)

        self.CHK_unroll_rotations = QtWidgets.QCheckBox("Unroll Rotations")
        self.LYT_options.addWidget(self.CHK_unroll_rotations)
        self.CHK_unroll_rotations.setChecked(True)
        # self.CHK_unroll_rotations.setEnabled(False)
        # self.CHK_bake_to_layer.toggled.connect(self.CHK_unroll_rotations.setEnabled)

    def _build_buttons_section(self):
        """Build the action buttons."""
        self.BTN_select_space = QtWidgets.QPushButton("Select Space Controls")
        self.LYT_main_vertical.addWidget(self.BTN_select_space)

        self.BTN_create_pins = QtWidgets.QPushButton("Create Pins")
        self.LYT_main_vertical.addWidget(self.BTN_create_pins)

        self.BTN_create_relative_pins = QtWidgets.QPushButton("Create Relative Pins")
        self.LYT_main_vertical.addWidget(self.BTN_create_relative_pins)

        self.BTN_bake_pins = QtWidgets.QPushButton("Bake Pins")
        self.LYT_main_vertical.addWidget(self.BTN_bake_pins)

    def _build_pin_list_section(self):
        """Build the pin group list widget."""
        self.LST_pin_groups = QtWidgets.QListWidget()
        self.LYT_main_vertical.addWidget(self.LST_pin_groups, 1)  # stretch factor of 1
        self.LST_pin_groups.setMinimumSize(0, 40)
        self.LST_pin_groups.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding)
        self.LST_pin_groups.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.LST_pin_groups.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.LST_pin_groups.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked |
            QtWidgets.QAbstractItemView.EditKeyPressed)
        self.LST_pin_groups.setAlternatingRowColors(True)
        self.LST_pin_groups.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def _build_context_menu(self):
        """Build the right-click context menu."""
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)

        self.popup_menu = QtWidgets.QMenu(self)
        self.menu_mini = QtWidgets.QAction('Miniaturize UI', self)
        self.popup_menu.addAction(self.menu_mini)
        self.menu_mini.triggered.connect(self.on_menu_mini_clicked)

    def _add_line_divider(self):
        """Add a horizontal line divider."""
        line = QtWidgets.QFrame()
        self.LYT_main_vertical.addWidget(line)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

    # Event handlers
    def mousePressEvent(self, event):
        self.pressPos = event.pos()
        self.isMoving = True

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

    def closeEvent(self, event):
        self._save_preferences()
        self.kill_callbacks()
        self.stop()

    # Callbacks
    @noUndo
    def init_callbacks(self):
        """Install Maya callbacks."""
        self.kill_callbacks(verbose=False)

        nullObj = api.MObject()
        self._callbacks['name_changed'] = api.MNodeMessage.addNameChangedCallback(
            nullObj, self._maya_node_name_changed)

        global MASTER_GROUP
        if not cmds.objExists(MASTER_GROUP):
            return

        master_group_sel = api.MGlobal.getSelectionListByName(MASTER_GROUP).getDagPath(0)

        self._callbacks[MASTER_GROUP + '_dag_added'] = \
            api.MDagMessage.addChildAddedDagPathCallback(
                master_group_sel, self._pin_group_child_added)

        self._callbacks[MASTER_GROUP + '_dag_removed'] = \
            api.MDagMessage.addChildRemovedDagPathCallback(
                master_group_sel, self._pin_group_child_removed)

    def kill_callbacks(self, verbose=True):
        """Remove Maya callbacks."""
        for ID in self._callbacks.keys():
            try:
                self._callbacks[ID] = api.MMessage.removeCallback(self._callbacks[ID]) or []
            except:
                pass

    def _pin_group_child_added(self, child, parent, client_data):
        cmds.evalDeferred(self._init_pin_group_list)

    def _pin_group_child_removed(self, child, parent, client_data):
        cmds.evalDeferred(self._init_pin_group_list)

    def _maya_node_name_changed(self, mObj, old_name, client_data):
        global MASTER_GROUP
        try:
            item_dag = api.MDagPath.getAPathTo(mObj)
        except:
            return

        if old_name == MASTER_GROUP:
            cmds.confirmDialog(
                backgroundColor=[0.85882, 0.19608, 0.03137],
                icon="critical",
                title='WAIT!',
                message='You renamed the animPin master group. This breaks everything.',
                button=["I promise I'll undo it!"])

        item_widget = self.LST_pin_groups.findItems(old_name, QtCore.Qt.MatchExactly)
        if item_widget:
            wigData = item_widget[0].data(QtCore.Qt.UserRole)
            if wigData == item_dag:
                widget_data_name = wigData.fullPathName().split('|')[-1]
                self.LST_pin_groups.blockSignals(True)
                item_widget[0].setText(widget_data_name)
                self.LST_pin_groups.blockSignals(False)

    def _ui_pin_name_changed(self, item):
        data = item.data(QtCore.Qt.UserRole)
        dataFN = api.MFnDependencyNode(data.node())
        self.LST_pin_groups.blockSignals(True)
        dataFN.setName(item.text())
        item.setText(data.fullPathName().split('|')[-1])
        self.LST_pin_groups.blockSignals(False)

    @noUndo
    def _init_pin_group_list(self):
        """Refresh the pin group list."""
        sel = cmds.ls(selection=True)
        self.LST_pin_groups.clear()
        pin_groups = get_pin_groups()

        if pin_groups:
            for pin_group in pin_groups:
                new_item = QtWidgets.QListWidgetItem(pin_group)
                mSel = api.MGlobal.getSelectionListByName(pin_group)
                item_dag = mSel.getDagPath(0)
                new_item.setData(QtCore.Qt.UserRole, item_dag)
                new_item.setFlags(
                    QtCore.Qt.ItemIsEditable |
                    QtCore.Qt.ItemIsSelectable |
                    QtCore.Qt.ItemIsEnabled)
                self.LST_pin_groups.addItem(new_item)
                if pin_group in sel:
                    new_item.setSelected(True)

    def _pass_selection_to_maya(self):
        selected_items = [x.text() for x in self.LST_pin_groups.selectedItems()]
        cmds.select(selected_items, replace=True)

    # Button handlers
    def on_select_space(self):
        try:
            from klugTools import rig_utils
            rig_utils.select_space_controls()
        except ImportError:
            api.MGlobal.displayWarning("klugTools.rig_utils not available")

    def on_create_pins(self):
        self.kill_callbacks(verbose=False)
        create_pins(
            start_frame=self.SPN_start_frame.value(),
            end_frame=self.SPN_end_frame.value())
        self.init_callbacks()
        self._init_pin_group_list()

    def on_create_relative_pins(self):
        self.kill_callbacks(verbose=False)
        create_relative_pins(
            start_frame=self.SPN_start_frame.value(),
            end_frame=self.SPN_end_frame.value())
        self.init_callbacks()
        self._init_pin_group_list()

    def on_bake_pins(self):
        option = 0 if self.OPT_0_matchkeys.isChecked() else self.SPN_step.value()
        bake_pins(
            bake_option=option,
            start_frame=self.SPN_start_frame.value(),
            end_frame=self.SPN_end_frame.value(),
            animLayer=self.CHK_bake_to_layer.isChecked(),
            unrollRotations=self.CHK_unroll_rotations.isChecked())
        self._init_pin_group_list()

    def on_context_menu(self, point):
        self.popup_menu.exec_(self.mapToGlobal(point))

    def on_menu_mini_clicked(self):
        if not self.mini_state:
            self.LST_pin_groups.hide()
            self.GRP_options.hide()
            self.LBL_start_frame.hide()
            self.LBL_end_frame.hide()
            self.SPN_start_frame.hide()
            self.SPN_end_frame.hide()
            self.width = self.geometry().width()
            self.height = self.geometry().height()
            self.mini_state = True
            self.menu_mini.setText("Embiggen UI")
        else:
            self.LST_pin_groups.show()
            self.GRP_options.show()
            self.LBL_start_frame.show()
            self.LBL_end_frame.show()
            self.SPN_start_frame.show()
            self.SPN_end_frame.show()
            self.mini_state = False
            self.menu_mini.setText("Miniaturize UI")
        QtCore.QTimer.singleShot(0, self.resize_window)

    def resize_window(self):
        if not self.mini_state:
            self.resize(self.width, self.height)
        else:
            self.resize(self.minimumSizeHint())


# =============================================================================
# UI Launcher Functions
# =============================================================================

def UI():
    """Launch the Animation Pin Tool UI."""
    AnimPinUI.run()


def show():
    """Alias for UI() - Launch the Animation Pin Tool."""
    AnimPinUI.run()


# =============================================================================
# Image Data (Base64 encoded header image)
# =============================================================================

IMAGE_DATA = base64.b64decode("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/GMzLdsAAEKeaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/Pgo8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA1LjYtYzA2NyA3OS4xNTc3NDcsIDIwMTUvMDMvMzAtMjM6NDA6NDIgICAgICAgICI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIKICAgICAgICAgICAgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIgogICAgICAgICAgICB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIKICAgICAgICAgICAgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIKICAgICAgICAgICAgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiCiAgICAgICAgICAgIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPHhtcDpDcmVhdG9yVG9vbD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoV2luZG93cyk8L3htcDpDcmVhdG9yVG9vbD4KICAgICAgICAgPHhtcDpDcmVhdGVEYXRlPjIwMTgtMDMtMTNUMDE6Mzc6MzAtMDc6MDA8L3htcDpDcmVhdGVEYXRlPgogICAgICAgICA8eG1wOk1ldGFkYXRhRGF0ZT4yMDE4LTA4LTMxVDE3OjA4OjQyLTA3OjAwPC94bXA6TWV0YWRhdGFEYXRlPgogICAgICAgICA8eG1wOk1vZGlmeURhdGU+MjAxOC0wOC0zMVQxNzowODo0Mi0wNzowMDwveG1wOk1vZGlmeURhdGU+CiAgICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2UvcG5nPC9kYzpmb3JtYXQ+CiAgICAgICAgIDx4bXBNTTpJbnN0YW5jZUlEPnhtcC5paWQ6ZjZiMWUxZWEtZDRlNC01NTRkLTkzZDYtYzUzNzlhODhiM2VhPC94bXBNTTpJbnN0YW5jZUlEPgogICAgICAgICA8eG1wTU06RG9jdW1lbnRJRD5hZG9iZTpkb2NpZDpwaG90b3Nob3A6MjM3MjgwN2MtYWQ3Yi0xMWU4LTgzNzQtZGI1M2Q5ZWQwNzNlPC94bXBNTTpEb2N1bWVudElEPgogICAgICAgICA8eG1wTU06T3JpZ2luYWxEb2N1bWVudElEPnhtcC5kaWQ6Nzk4MTg5ZDUtMDFhYi04NDRlLTlhMDUtOWU0ZmFiMDNiYzFmPC94bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ+CiAgICAgICAgIDx4bXBNTTpIaXN0b3J5PgogICAgICAgICAgICA8cmRmOlNlcT4KICAgICAgICAgICAgICAgPHJkZjpsaSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDphY3Rpb24+Y3JlYXRlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6aW5zdGFuY2VJRD54bXAuaWlkOjc5ODE4OWQ1LTAxYWItODQ0ZS05YTA1LTllNGZhYjAzYmMxZjwvc3RFdnQ6aW5zdGFuY2VJRD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OndoZW4+MjAxOC0wMy0xM1QwMTozNzozMC0wNzowMDwvc3RFdnQ6d2hlbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnNvZnR3YXJlQWdlbnQ+QWRvYmUgUGhvdG9zaG9wIENDIDIwMTUgKFdpbmRvd3MpPC9zdEV2dDpzb2Z0d2FyZUFnZW50PgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgICAgPHJkZjpsaSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDphY3Rpb24+c2F2ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0Omluc3RhbmNlSUQ+eG1wLmlpZDpiMWFhNjVjNC1mZGRkLThhNDMtOGFlNC0yMzViOGY4NmQwYzg8L3N0RXZ0Omluc3RhbmNlSUQ+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDp3aGVuPjIwMTgtMDMtMTNUMDE6Mzk6NTUtMDc6MDA8L3N0RXZ0OndoZW4+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDpzb2Z0d2FyZUFnZW50PkFkb2JlIFBob3Rvc2hvcCBDQyAyMDE1IChXaW5kb3dzKTwvc3RFdnQ6c29mdHdhcmVBZ2VudD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmNoYW5nZWQ+Lzwvc3RFdnQ6Y2hhbmdlZD4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPnNhdmVkPC9zdEV2dDphY3Rpb24+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDppbnN0YW5jZUlEPnhtcC5paWQ6MWM2MjhjM2UtZjYxZi1lYTRmLWI0NzgtNzViNGU1ODM4Nzc4PC9zdEV2dDppbnN0YW5jZUlEPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6d2hlbj4yMDE4LTA4LTMxVDE3OjA4OjQyLTA3OjAwPC9zdEV2dDp3aGVuPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6c29mdHdhcmVBZ2VudD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoV2luZG93cyk8L3N0RXZ0OnNvZnR3YXJlQWdlbnQ+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDpjaGFuZ2VkPi88L3N0RXZ0OmNoYW5nZWQ+CiAgICAgICAgICAgICAgIDwvcmRmOmxpPgogICAgICAgICAgICAgICA8cmRmOmxpIHJkZjpwYXJzZVR5cGU9IlJlc291cmNlIj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmFjdGlvbj5jb252ZXJ0ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnBhcmFtZXRlcnM+ZnJvbSBhcHBsaWNhdGlvbi92bmQuYWRvYmUucGhvdG9zaG9wIHRvIGltYWdlL3BuZzwvc3RFdnQ6cGFyYW1ldGVycz4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPmRlcml2ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnBhcmFtZXRlcnM+Y29udmVydGVkIGZyb20gYXBwbGljYXRpb24vdm5kLmFkb2JlLnBob3Rvc2hvcCB0byBpbWFnZS9wbmc8L3N0RXZ0OnBhcmFtZXRlcnM+CiAgICAgICAgICAgICAgIDwvcmRmOmxpPgogICAgICAgICAgICAgICA8cmRmOmxpIHJkZjpwYXJzZVR5cGU9IlJlc291cmNlIj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmFjdGlvbj5zYXZlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6aW5zdGFuY2VJRD54bXAuaWlkOmY2YjFlMWVhLWQ0ZTQtNTU0ZC05M2Q2LWM1Mzc5YTg4YjNlYTwvc3RFdnQ6aW5zdGFuY2VJRD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OndoZW4+MjAxOC0wOC0zMVQxNzowODo0Mi0wNzowMDwvc3RFdnQ6d2hlbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnNvZnR3YXJlQWdlbnQ+QWRvYmUgUGhvdG9zaG9wIENDIDIwMTUgKFdpbmRvd3MpPC9zdEV2dDpzb2Z0d2FyZUFnZW50PgogICAgICAgICAgICAgICAgICA8c3RFdnQ6Y2hhbmdlZD4vPC9zdEV2dDpjaGFuZ2VkPgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgPC9yZGY6U2VxPgogICAgICAgICA8L3htcE1NOkhpc3Rvcnk+CiAgICAgICAgIDx4bXBNTTpEZXJpdmVkRnJvbSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgIDxzdFJlZjppbnN0YW5jZUlEPnhtcC5paWQ6MWM2MjhjM2UtZjYxZi1lYTRmLWI0NzgtNzViNGU1ODM4Nzc4PC9zdFJlZjppbnN0YW5jZUlEPgogICAgICAgICAgICA8c3RSZWY6ZG9jdW1lbnRJRD5hZG9iZTpkb2NpZDpwaG90b3Nob3A6ZDAxMjIxZjktM2JjNy0xMWU4LThjZmItZTllNzY2YThmODk1PC9zdFJlZjpkb2N1bWVudElEPgogICAgICAgICAgICA8c3RSZWY6b3JpZ2luYWxEb2N1bWVudElEPnhtcC5kaWQ6Nzk4MTg5ZDUtMDFhYi04NDRlLTlhMDUtOWU0ZmFiMDNiYzFmPC9zdFJlZjpvcmlnaW5hbERvY3VtZW50SUQ+CiAgICAgICAgIDwveG1wTU06RGVyaXZlZEZyb20+CiAgICAgICAgIDxwaG90b3Nob3A6Q29sb3JNb2RlPjM8L3Bob3Rvc2hvcDpDb2xvck1vZGU+CiAgICAgICAgIDxwaG90b3Nob3A6SUNDUHJvZmlsZT5zUkdCIElFQzYxOTY2LTIuMTwvcGhvdG9zaG9wOklDQ1Byb2ZpbGU+CiAgICAgICAgIDxwaG90b3Nob3A6VGV4dExheWVycz4KICAgICAgICAgICAgPHJkZjpCYWc+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8cGhvdG9zaG9wOkxheWVyTmFtZT5hbmltUGluPC9waG90b3Nob3A6TGF5ZXJOYW1lPgogICAgICAgICAgICAgICAgICA8cGhvdG9zaG9wOkxheWVyVGV4dD5hbmltUGluPC9waG90b3Nob3A6TGF5ZXJUZXh0PgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgPC9yZGY6QmFnPgogICAgICAgICA8L3Bob3Rvc2hvcDpUZXh0TGF5ZXJzPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICAgICA8dGlmZjpYUmVzb2x1dGlvbj4zMDAwMDAwLzEwMDAwPC90aWZmOlhSZXNvbHV0aW9uPgogICAgICAgICA8dGlmZjpZUmVzb2x1dGlvbj4zMDAwMDAwLzEwMDAwPC90aWZmOllSZXNvbHV0aW9uPgogICAgICAgICA8dGlmZjpSZXNvbHV0aW9uVW5pdD4yPC90aWZmOlJlc29sdXRpb25Vbml0PgogICAgICAgICA8ZXhpZjpDb2xvclNwYWNlPjE8L2V4aWY6Q29sb3JTcGFjZT4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjE4MDwvZXhpZjpQaXhlbFhEaW1lbnNpb24+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj41MDwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo8P3hwYWNrZXQgZW5kPSJ3Ij8+dyLjjQAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAA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")

# Eof

