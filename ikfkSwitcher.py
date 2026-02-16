# IK/FK Switcher Tool
# -------------------------------------------------------------------- #

"""
IK/FK Switcher Tool

A companion tool to the Animation Pin Tool that enables IK/FK switching
for spine rigs (and similar setups). Uses the pin tool's world-space
capture/restore workflow to stabilize IK controls while matching FK
controls to the deformation skeleton.

Usage:
    import ikfkSwitcher
    ikfkSwitcher.show()
"""

__author__ = "Daniel Klug"
__version__ = "1.0.0"
__date__ = "2026-02-14"

# =============================================================================
# Imports
# =============================================================================

import json
import os
import logging

import maya.cmds as cmds
import maya.OpenMayaUI as mui

from Qt import QtGui, QtCore, QtCompat, QtWidgets

import animPin

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# =============================================================================
# Constants
# =============================================================================

WIDTH = 340
HEIGHT = 620

PRESETS_FILENAME = "ikfk_presets.json"
PRESETS_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_PATH = os.path.join(PRESETS_DIR, PRESETS_FILENAME)

# =============================================================================
# Helpers
# =============================================================================

def _get_maya_window():
    """Get the main Maya window as a Qt widget."""
    ptr = mui.MQtUtil.mainWindow()
    return QtCompat.wrapInstance(int(ptr), QtWidgets.QMainWindow)


def _load_presets_from_disk():
    """Load all presets from the JSON file on disk."""
    if not os.path.isfile(PRESETS_PATH):
        return {}
    try:
        with open(PRESETS_PATH, 'r') as f:
            return json.load(f)
    except (IOError, ValueError):
        log.warning("Failed to read presets file: %s", PRESETS_PATH)
        return {}


def _save_presets_to_disk(presets):
    """Save all presets to the JSON file on disk."""
    try:
        with open(PRESETS_PATH, 'w') as f:
            json.dump(presets, f, indent=2, sort_keys=True)
    except IOError:
        log.error("Failed to write presets file: %s", PRESETS_PATH)


# =============================================================================
# IK/FK Switch Operations
# =============================================================================

def _validate_scene_nodes(fk_mapping, ik_controls, blend_attr):
    """Validate that all referenced nodes/attributes exist in the scene.

    Returns:
        list: Error messages. Empty if everything is valid.
    """
    errors = []
    for ctrl, jnt in fk_mapping:
        if not cmds.objExists(ctrl):
            errors.append("FK control not found: {}".format(ctrl))
        if not cmds.objExists(jnt):
            errors.append("Joint not found: {}".format(jnt))
    for ctrl in ik_controls:
        if not cmds.objExists(ctrl):
            errors.append("IK control not found: {}".format(ctrl))
    if blend_attr and not cmds.objExists(blend_attr):
        errors.append("Blend attribute not found: {}".format(blend_attr))
    return errors


def _create_temp_locator(ctrl):
    """Create a temporary locator matched to a control's world-space transform."""
    short = ctrl.split('|')[-1].split(':')[-1]
    loc = cmds.spaceLocator(name=short + '_ikfk_tmp')[0]
    ro = cmds.getAttr(ctrl + '.rotateOrder')
    cmds.setAttr(loc + '.rotateOrder', ro)
    # Snap locator to control via temp constraint
    cmds.delete(cmds.parentConstraint(ctrl, loc))
    return loc


def _do_bake_transforms(nodes, start_frame, end_frame, sample=1):
    """Bake transform attributes only (tx ty tz rx ry rz)."""
    cmds.bakeResults(
        nodes,
        simulation=True,
        time=(start_frame, end_frame),
        sampleBy=sample,
        oversamplingRate=1,
        disableImplicitControl=True,
        preserveOutsideKeys=True,
        sparseAnimCurveBake=False,
        removeBakedAttributeFromLayer=False,
        removeBakedAnimFromLayer=False,
        bakeOnOverrideLayer=False,
        minimizeRotation=True,
        controlPoints=False,
        shape=True,
        at=("tx", "ty", "tz", "rx", "ry", "rz"))


def _pin_ik_controls(ik_controls, start_frame, end_frame):
    """Pin IK controls by creating baked locators that hold world-space positions.

    Creates a locator per IK control, bakes the control's world-space motion
    onto it, then reverses the constraint so the locator drives the control.

    Returns:
        tuple: (locators, pin_constraints) for cleanup later.
    """
    locators = []
    capture_constraints = []

    for ctrl in ik_controls:
        loc = _create_temp_locator(ctrl)
        locators.append(loc)
        con = cmds.parentConstraint(ctrl, loc)[0]
        capture_constraints.append(con)

    # Bake locators to capture world-space motion
    _do_bake_transforms(locators, start_frame, end_frame)

    # Remove capture constraints
    cmds.delete(capture_constraints)

    # Reverse: constrain IK controls to the baked locators
    pin_constraints = []
    for ctrl, loc in zip(ik_controls, locators):
        con = cmds.parentConstraint(loc, ctrl, mo=False)[0]
        pin_constraints.append(con)

    return locators, pin_constraints


def _zero_controls(controls, start_frame, end_frame):
    """Zero out translate and rotate animation on the given controls.

    Cuts all keys in the frame range and sets tx/ty/tz/rx/ry/rz to 0.
    """
    attrs = ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')
    for ctrl in controls:
        for attr in attrs:
            plug = '{}.{}'.format(ctrl, attr)
            if not cmds.objExists(plug):
                continue
            if cmds.getAttr(plug, lock=True):
                continue
            cmds.cutKey(plug, time=(start_frame, end_frame), clear=True)
            cmds.setAttr(plug, 0)


def _set_blend_attr(blend_attr, value, start_frame, end_frame):
    """Set the IK/FK blend attribute to a value across the frame range."""
    if not blend_attr or not cmds.objExists(blend_attr):
        return
    cmds.cutKey(blend_attr, time=(start_frame, end_frame), clear=True)
    cmds.setKeyframe(blend_attr, time=start_frame, value=value)
    cmds.setKeyframe(blend_attr, time=end_frame, value=value)


@animPin.undo
@animPin.viewport_off
def switch_to_fk(fk_mapping, ik_controls, blend_attr, fk_value,
                 start_frame, end_frame, sample=1,
                 animLayer=False, unrollRotations=True):
    """Switch from IK to FK mode.

    Workflow:
        1. Pin IK controls (stabilize while FK controls are matched)
        2. Constrain FK controls to their deformation joints
        3. Bake FK controls + IK controls in one pass
        4. Clean up constraints and locators
        5. Set blend attribute to FK value
    """
    fk_controls = [pair[0] for pair in fk_mapping]
    all_controls = fk_controls + list(ik_controls)

    # Bookend curves to preserve animation outside the work area
    animPin.bookend_curves(all_controls, start_frame, end_frame)

    # Step 1: Pin IK controls
    locators, ik_pin_constraints = _pin_ik_controls(
        ik_controls, start_frame, end_frame)

    # Step 2: Constrain FK controls to deformation joints
    fk_constraints = []
    for ctrl, jnt in fk_mapping:
        con = animPin.parent_constraint_with_skips(jnt, ctrl)
        if con:
            fk_constraints.append(con)

    # Step 3: Bake everything in one pass
    all_constraints = fk_constraints + ik_pin_constraints
    if animLayer:
        animPin.do_bake_to_layer(
            all_controls, start_frame, end_frame, sample,
            unrollRotations=unrollRotations,
            constraints=all_constraints)
    else:
        _do_bake_transforms(all_controls, start_frame, end_frame, sample)
        remaining = [c for c in all_constraints if cmds.objExists(c)]
        if remaining:
            cmds.delete(remaining)

    # Step 4: Delete pin locators
    remaining_locs = [loc for loc in locators if cmds.objExists(loc)]
    if remaining_locs:
        cmds.delete(remaining_locs)

    # Step 5: Set blend to FK
    _set_blend_attr(blend_attr, fk_value, start_frame, end_frame)

    print("IK/FK Switcher: Switched to FK mode.")


@animPin.undo
@animPin.viewport_off
def switch_to_ik(ik_controls, blend_attr, ik_value,
                 start_frame, end_frame, sample=1,
                 animLayer=False, unrollRotations=True,
                 reset_controls=None):
    """Switch from FK to IK mode.

    Workflow:
        1. Pin IK controls (captures world-space positions from FK-driven rig)
        2. Zero out IK controls that need resetting (clears stale IK animation)
        3. Bake IK controls from pins
        4. Clean up constraints and locators
        5. Set blend attribute to IK value

    Args:
        reset_controls: List of IK control names to zero out before baking.
            Controls NOT in this list (e.g. pelvis) keep their animation.
    """
    # Bookend curves to preserve animation outside the work area
    animPin.bookend_curves(list(ik_controls), start_frame, end_frame)

    # Step 1: Pin IK controls (captures current world-space from FK-driven rig)
    locators, ik_pin_constraints = _pin_ik_controls(
        ik_controls, start_frame, end_frame)

    # Step 2: Zero out stale IK animation on controls marked for reset
    if reset_controls:
        _zero_controls(reset_controls, start_frame, end_frame)

    # Step 3: Bake IK controls
    if animLayer:
        animPin.do_bake_to_layer(
            list(ik_controls), start_frame, end_frame, sample,
            unrollRotations=unrollRotations,
            constraints=ik_pin_constraints)
    else:
        _do_bake_transforms(list(ik_controls), start_frame, end_frame, sample)
        remaining = [c for c in ik_pin_constraints if cmds.objExists(c)]
        if remaining:
            cmds.delete(remaining)

    # Step 4: Delete pin locators
    remaining_locs = [loc for loc in locators if cmds.objExists(loc)]
    if remaining_locs:
        cmds.delete(remaining_locs)

    # Step 5: Set blend to IK
    _set_blend_attr(blend_attr, ik_value, start_frame, end_frame)

    print("IK/FK Switcher: Switched to IK mode.")


# =============================================================================
# UI
# =============================================================================

class IKFKSwitcherUI(QtWidgets.QDialog):
    """IK/FK Switcher Tool UI"""

    UI_INSTANCE = None

    @classmethod
    def _is_valid_widget(cls, widget):
        if widget is None:
            return False
        return QtCompat.isValid(widget)

    @classmethod
    def run(cls):
        """Show the UI. Creates a new instance if needed."""
        print("IK/FK Switcher version: {}".format(__version__))
        if cls.UI_INSTANCE is not None and not cls._is_valid_widget(cls.UI_INSTANCE):
            cls.UI_INSTANCE = None

        if cls.UI_INSTANCE is None:
            cls.UI_INSTANCE = IKFKSwitcherUI()

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
        super(IKFKSwitcherUI, self).__init__(parent)
        self.parent = _get_maya_window()
        self.setParent(self.parent)
        self.setWindowFlags(
            QtCore.Qt.Dialog |
            QtCore.Qt.WindowCloseButtonHint)
        self.setObjectName('IKFKSwitcher')
        self.setWindowTitle('IK/FK Switcher')
        self.setProperty("saveWindowPref", True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self.pressPos = None
        self.isMoving = False

        self.build_UI()
        self._populate_preset_dropdown()
        self.init_frame_range()
        self.init_connections()

    # -----------------------------------------------------------------
    # Connections
    # -----------------------------------------------------------------

    def init_connections(self):
        """Set up signal/slot connections."""
        self.CMB_preset.currentIndexChanged.connect(self._on_preset_selected)
        self.BTN_save_preset.clicked.connect(self._on_save_preset)
        self.BTN_delete_preset.clicked.connect(self._on_delete_preset)

        self.BTN_fk_add_row.clicked.connect(self._on_fk_add_row)
        self.BTN_fk_remove_row.clicked.connect(self._on_fk_remove_row)
        self.BTN_fk_load_sel.clicked.connect(self._on_fk_load_selection)

        self.BTN_ik_add.clicked.connect(self._on_ik_add)
        self.BTN_ik_remove.clicked.connect(self._on_ik_remove)
        self.BTN_ik_load_sel.clicked.connect(self._on_ik_load_selection)

        self.BTN_switch_to_fk.clicked.connect(self._on_switch_to_fk)
        self.BTN_switch_to_ik.clicked.connect(self._on_switch_to_ik)

    def init_frame_range(self):
        """Initialize frame range from timeline."""
        start = cmds.playbackOptions(query=True, animationStartTime=True)
        end = cmds.playbackOptions(query=True, animationEndTime=True)
        self.SPN_start_frame.setValue(start)
        self.SPN_end_frame.setValue(end)

    # -----------------------------------------------------------------
    # Build UI
    # -----------------------------------------------------------------

    def build_UI(self):
        self._apply_stylesheet()

        self.LYT_main = QtWidgets.QVBoxLayout()
        self.setLayout(self.LYT_main)
        self.LYT_main.setContentsMargins(10, 10, 10, 10)
        self.LYT_main.setSpacing(8)

        self._build_preset_section()
        self._add_line_divider()
        self._build_fk_mapping_section()
        self._add_line_divider()
        self._build_ik_controls_section()
        self._add_line_divider()
        self._build_blend_attr_section()
        self._add_line_divider()
        self._build_frame_range_section()
        self._add_line_divider()
        self._build_bake_options_section()
        self._add_line_divider()
        self._build_action_buttons()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
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
                color: rgb(120, 120, 120);
                subcontrol-origin: margin;
                subcontrol-position: top center;
                margin: 0px 4px;
                padding: 0px;
            }
            QLabel#headerLabel {
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
            QPushButton {
                background-color: rgb(80, 80, 80);
                border-style: solid;
                border-width: 0px;
                border-radius: 8px;
                color: rgb(186, 186, 186);
                min-height: 28px;
            }
            QPushButton:hover { background-color: rgb(90, 90, 90); }
            QPushButton:pressed { background-color: rgb(74, 105, 129); }
            QPushButton#actionButton {
                min-height: 50px;
                font: bold 11pt Sans-serif;
            }
            QTableWidget {
                background: rgb(65, 65, 65);
                border: 1px solid grey;
                border-radius: 6px;
                border-color: rgb(80, 80, 80);
                gridline-color: rgb(80, 80, 80);
            }
            QTableWidget::item {
                background: rgb(65, 65, 65);
                padding: 2px 4px;
            }
            QTableWidget::item:selected { background-color: #4a6981; }
            QTableWidget::item:hover { background: rgb(80, 80, 80); }
            QHeaderView::section {
                background-color: rgb(60, 60, 60);
                color: rgb(160, 160, 160);
                padding: 4px;
                border: 1px solid rgb(80, 80, 80);
                font: bold 9pt Sans-serif;
            }
            QListWidget {
                show-decoration-selected: 1;
                background: rgb(65, 65, 65);
                border: 1px solid grey;
                border-radius: 6px;
                padding: 4px;
                border-color: rgb(80, 80, 80);
            }
            QListWidget::item {
                background: rgb(65, 65, 65);
                margin-bottom: 2px;
                border-radius: 4px;
                padding-left: 4px;
                height: 22px;
            }
            QListWidget::item:selected { background-color: #4a6981; }
            QListWidget::item:hover { background: rgb(80, 80, 80); }
            QLineEdit {
                background-color: rgb(50, 50, 50);
                border: 1px solid rgb(80, 80, 80);
                border-radius: 4px;
                color: rgb(180, 180, 180);
                padding: 2px 4px;
            }
            QComboBox {
                background-color: rgb(60, 60, 60);
                border: 1px solid rgb(80, 80, 80);
                border-radius: 4px;
                color: rgb(180, 180, 180);
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: rgb(60, 60, 60);
                border: 1px solid rgb(80, 80, 80);
                selection-background-color: #4a6981;
                color: rgb(180, 180, 180);
            }
            QScrollBar:vertical {
                border: 1px solid grey;
                border-color: rgb(90, 90, 90);
                background: rgb(60, 60, 60);
                width: 8px;
            }
            QScrollBar::handle:vertical { background: rgb(90, 90, 90); min-height: 20px; }
            QRadioButton {
                background-color: rgb(65, 65, 65);
                color: rgb(180, 180, 180);
                border-radius: 8px;
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

    # -- Preset section --

    def _build_preset_section(self):
        lyt = QtWidgets.QHBoxLayout()
        self.LYT_main.addLayout(lyt)

        lbl = QtWidgets.QLabel("Preset:")
        lbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        lyt.addWidget(lbl)

        self.CMB_preset = QtWidgets.QComboBox()
        self.CMB_preset.setMinimumWidth(120)
        self.CMB_preset.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lyt.addWidget(self.CMB_preset)

        self.BTN_save_preset = QtWidgets.QPushButton("Save")
        self.BTN_save_preset.setMaximumWidth(60)
        lyt.addWidget(self.BTN_save_preset)

        self.BTN_delete_preset = QtWidgets.QPushButton("Delete")
        self.BTN_delete_preset.setMaximumWidth(60)
        lyt.addWidget(self.BTN_delete_preset)

    # -- FK mapping table --

    def _build_fk_mapping_section(self):
        grp = QtWidgets.QGroupBox("FK Mapping")
        self.LYT_main.addWidget(grp)
        lyt = QtWidgets.QVBoxLayout()
        grp.setLayout(lyt)
        lyt.setContentsMargins(8, 16, 8, 8)
        lyt.setSpacing(6)

        self.TBL_fk_mapping = QtWidgets.QTableWidget()
        self.TBL_fk_mapping.setColumnCount(2)
        self.TBL_fk_mapping.setHorizontalHeaderLabels(["FK Control", "Joint"])
        self.TBL_fk_mapping.horizontalHeader().setStretchLastSection(True)
        self.TBL_fk_mapping.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        self.TBL_fk_mapping.verticalHeader().setVisible(False)
        self.TBL_fk_mapping.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.TBL_fk_mapping.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.TBL_fk_mapping.setMinimumHeight(100)
        self.TBL_fk_mapping.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding)
        lyt.addWidget(self.TBL_fk_mapping)

        btn_lyt = QtWidgets.QHBoxLayout()
        lyt.addLayout(btn_lyt)

        self.BTN_fk_add_row = QtWidgets.QPushButton("+ Add Row")
        btn_lyt.addWidget(self.BTN_fk_add_row)

        self.BTN_fk_remove_row = QtWidgets.QPushButton("- Remove Row")
        btn_lyt.addWidget(self.BTN_fk_remove_row)

        self.BTN_fk_load_sel = QtWidgets.QPushButton("<< Load Sel")
        btn_lyt.addWidget(self.BTN_fk_load_sel)

    # -- IK controls list --

    def _build_ik_controls_section(self):
        grp = QtWidgets.QGroupBox("IK Controls (to pin)")
        self.LYT_main.addWidget(grp)
        lyt = QtWidgets.QVBoxLayout()
        grp.setLayout(lyt)
        lyt.setContentsMargins(8, 16, 8, 8)
        lyt.setSpacing(6)

        self.LST_ik_controls = QtWidgets.QListWidget()
        self.LST_ik_controls.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.LST_ik_controls.setMinimumHeight(60)
        self.LST_ik_controls.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding)
        lyt.addWidget(self.LST_ik_controls)

        btn_lyt = QtWidgets.QHBoxLayout()
        lyt.addLayout(btn_lyt)

        self.BTN_ik_add = QtWidgets.QPushButton("+ Add")
        btn_lyt.addWidget(self.BTN_ik_add)

        self.BTN_ik_remove = QtWidgets.QPushButton("- Remove")
        btn_lyt.addWidget(self.BTN_ik_remove)

        self.BTN_ik_load_sel = QtWidgets.QPushButton("<< Load Sel")
        btn_lyt.addWidget(self.BTN_ik_load_sel)

    # -- Blend attribute --

    def _build_blend_attr_section(self):
        lyt = QtWidgets.QGridLayout()
        self.LYT_main.addLayout(lyt)
        lyt.setContentsMargins(0, 0, 0, 0)

        lbl_attr = QtWidgets.QLabel("Blend Attr:")
        lyt.addWidget(lbl_attr, 0, 0)
        self.TXT_blend_attr = QtWidgets.QLineEdit()
        self.TXT_blend_attr.setPlaceholderText("e.g. spine_settings.ikfk_blend")
        lyt.addWidget(self.TXT_blend_attr, 0, 1, 1, 3)

        lbl_fk_val = QtWidgets.QLabel("FK Value:")
        lyt.addWidget(lbl_fk_val, 1, 0)
        self.SPN_fk_value = QtWidgets.QSpinBox()
        self.SPN_fk_value.setRange(0, 10)
        self.SPN_fk_value.setValue(0)
        self.SPN_fk_value.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_fk_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        lyt.addWidget(self.SPN_fk_value, 1, 1)

        lbl_ik_val = QtWidgets.QLabel("IK Value:")
        lyt.addWidget(lbl_ik_val, 1, 2)
        self.SPN_ik_value = QtWidgets.QSpinBox()
        self.SPN_ik_value.setRange(0, 10)
        self.SPN_ik_value.setValue(1)
        self.SPN_ik_value.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_ik_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        lyt.addWidget(self.SPN_ik_value, 1, 3)

    # -- Frame range --

    def _build_frame_range_section(self):
        lyt = QtWidgets.QGridLayout()
        self.LYT_main.addLayout(lyt)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setColumnStretch(1, 2)

        lbl_start = QtWidgets.QLabel("Start Frame")
        lyt.addWidget(lbl_start, 0, 0)

        self.SPN_start_frame = QtWidgets.QSpinBox()
        lyt.addWidget(self.SPN_start_frame, 0, 1)
        self.SPN_start_frame.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_start_frame.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_start_frame.setRange(-999999999, 999999999)

        lbl_end = QtWidgets.QLabel("End Frame")
        lyt.addWidget(lbl_end, 1, 0)

        self.SPN_end_frame = QtWidgets.QSpinBox()
        lyt.addWidget(self.SPN_end_frame, 1, 1)
        self.SPN_end_frame.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_end_frame.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_end_frame.setRange(-999999999, 999999999)

    # -- Bake options --

    def _build_bake_options_section(self):
        grp = QtWidgets.QGroupBox("Bake Options")
        self.LYT_main.addWidget(grp)
        grp.setMinimumHeight(100)
        grp.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Minimum)

        lyt = QtWidgets.QVBoxLayout()
        grp.setLayout(lyt)
        lyt.setSpacing(6)
        lyt.setContentsMargins(10, 12, 10, 4)

        self.OPT_match_keys = QtWidgets.QRadioButton("Match Keys")
        lyt.addWidget(self.OPT_match_keys)
        self.OPT_match_keys.setChecked(True)

        step_lyt = QtWidgets.QHBoxLayout()
        lyt.addLayout(step_lyt)

        self.OPT_bake_step = QtWidgets.QRadioButton("Bake step ")
        self.OPT_bake_step.setMinimumWidth(self.OPT_bake_step.sizeHint().width())
        self.OPT_bake_step.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed)
        step_lyt.addWidget(self.OPT_bake_step, 0)

        self.SPN_step = QtWidgets.QSpinBox()
        self.SPN_step.setMinimumWidth(40)
        step_lyt.addWidget(self.SPN_step, 1)
        self.SPN_step.setAlignment(QtCore.Qt.AlignRight)
        self.SPN_step.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.SPN_step.setRange(1, 999999999)
        self.SPN_step.setValue(2)

        self.CHK_bake_to_layer = QtWidgets.QCheckBox("Bake to Animation Layer")
        lyt.addWidget(self.CHK_bake_to_layer)

        self.CHK_unroll_rotations = QtWidgets.QCheckBox("Unroll Rotations")
        lyt.addWidget(self.CHK_unroll_rotations)
        self.CHK_unroll_rotations.setChecked(True)

    # -- Action buttons --

    def _build_action_buttons(self):
        self.BTN_switch_to_fk = QtWidgets.QPushButton("Switch to FK")
        self.BTN_switch_to_fk.setObjectName("actionButton")
        self.LYT_main.addWidget(self.BTN_switch_to_fk)

        self.BTN_switch_to_ik = QtWidgets.QPushButton("Switch to IK")
        self.BTN_switch_to_ik.setObjectName("actionButton")
        self.LYT_main.addWidget(self.BTN_switch_to_ik)

    # -- Divider helper --

    def _add_line_divider(self):
        line = QtWidgets.QFrame()
        self.LYT_main.addWidget(line)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

    # -----------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------

    def mousePressEvent(self, event):
        self.pressPos = event.pos()
        self.isMoving = True

    def mouseReleaseEvent(self, event):
        self.isMoving = False

    def mouseMoveEvent(self, event):
        if self.isMoving:
            newPos = event.pos() - self.pressPos
            self.move(self.window().pos() + newPos)

    def showEvent(self, event):
        self.resize(WIDTH, HEIGHT)

    def closeEvent(self, event):
        self.stop()

    # -----------------------------------------------------------------
    # Preset management
    # -----------------------------------------------------------------

    def _populate_preset_dropdown(self):
        """Refresh the preset combobox from disk."""
        self.CMB_preset.blockSignals(True)
        self.CMB_preset.clear()
        self.CMB_preset.addItem("(none)")
        presets = _load_presets_from_disk()
        for name in sorted(presets.keys()):
            self.CMB_preset.addItem(name)
        self.CMB_preset.blockSignals(False)

    def _on_preset_selected(self, index):
        """Load the selected preset into the UI."""
        if index <= 0:
            return
        name = self.CMB_preset.currentText()
        presets = _load_presets_from_disk()
        data = presets.get(name)
        if not data:
            return
        self._apply_preset_data(data)

    def _apply_preset_data(self, data):
        """Populate UI fields from a preset dictionary."""
        # FK mapping
        fk_mapping = data.get("fk_mapping", [])
        self.TBL_fk_mapping.setRowCount(0)
        for pair in fk_mapping:
            row = self.TBL_fk_mapping.rowCount()
            self.TBL_fk_mapping.insertRow(row)
            self.TBL_fk_mapping.setItem(
                row, 0, QtWidgets.QTableWidgetItem(pair[0]))
            self.TBL_fk_mapping.setItem(
                row, 1, QtWidgets.QTableWidgetItem(pair[1]))

        # IK controls (supports both old format ["name"] and new [["name", reset]])
        ik_controls = data.get("ik_controls", [])
        self.LST_ik_controls.clear()
        for entry in ik_controls:
            if isinstance(entry, list):
                name, reset = entry[0], entry[1]
            else:
                name, reset = entry, True
            self._add_ik_item(name, reset=reset)

        # Blend attribute
        self.TXT_blend_attr.setText(data.get("blend_attr", ""))
        self.SPN_fk_value.setValue(data.get("fk_value", 0))
        self.SPN_ik_value.setValue(data.get("ik_value", 1))

    def _collect_preset_data(self):
        """Gather current UI state into a preset dictionary."""
        fk_mapping = []
        for row in range(self.TBL_fk_mapping.rowCount()):
            ctrl_item = self.TBL_fk_mapping.item(row, 0)
            jnt_item = self.TBL_fk_mapping.item(row, 1)
            ctrl = ctrl_item.text() if ctrl_item else ""
            jnt = jnt_item.text() if jnt_item else ""
            fk_mapping.append([ctrl, jnt])

        ik_controls = []
        for i in range(self.LST_ik_controls.count()):
            item = self.LST_ik_controls.item(i)
            reset = item.checkState() == QtCore.Qt.Checked
            ik_controls.append([item.text(), reset])

        return {
            "fk_mapping": fk_mapping,
            "ik_controls": ik_controls,
            "blend_attr": self.TXT_blend_attr.text(),
            "fk_value": self.SPN_fk_value.value(),
            "ik_value": self.SPN_ik_value.value(),
        }

    def _on_save_preset(self):
        """Save the current configuration as a named preset."""
        current = self.CMB_preset.currentText()
        default_name = current if current != "(none)" else ""

        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            QtWidgets.QLineEdit.Normal, default_name)
        if not ok or not name.strip():
            return

        name = name.strip()
        presets = _load_presets_from_disk()
        presets[name] = self._collect_preset_data()
        _save_presets_to_disk(presets)
        self._populate_preset_dropdown()

        # Select the newly saved preset
        idx = self.CMB_preset.findText(name)
        if idx >= 0:
            self.CMB_preset.setCurrentIndex(idx)

    def _on_delete_preset(self):
        """Delete the currently selected preset."""
        name = self.CMB_preset.currentText()
        if name == "(none)":
            return

        result = QtWidgets.QMessageBox.question(
            self, "Delete Preset",
            "Delete preset '{}'?".format(name),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if result != QtWidgets.QMessageBox.Yes:
            return

        presets = _load_presets_from_disk()
        presets.pop(name, None)
        _save_presets_to_disk(presets)
        self._populate_preset_dropdown()

    # -----------------------------------------------------------------
    # FK mapping table handlers
    # -----------------------------------------------------------------

    def _on_fk_add_row(self):
        """Add an empty row to the FK mapping table."""
        row = self.TBL_fk_mapping.rowCount()
        self.TBL_fk_mapping.insertRow(row)
        self.TBL_fk_mapping.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.TBL_fk_mapping.setItem(row, 1, QtWidgets.QTableWidgetItem(""))

    def _on_fk_remove_row(self):
        """Remove selected rows from the FK mapping table."""
        rows = sorted(set(idx.row() for idx in self.TBL_fk_mapping.selectedIndexes()),
                       reverse=True)
        for row in rows:
            self.TBL_fk_mapping.removeRow(row)

    def _on_fk_load_selection(self):
        """Load Maya selection into the FK mapping table.

        Smart behavior:
        - 1 object: fills the currently focused cell
        - 2+ objects: separates joints from transforms, pairs them,
          and creates new rows
        """
        sel = cmds.ls(selection=True, long=True)
        if not sel:
            cmds.warning("Nothing selected.")
            return

        if len(sel) == 1:
            # Single selection - fill the active cell
            self._fill_active_fk_cell(sel[0])
            return

        # Multiple selection - sort by type and pair
        joints = []
        controls = []
        for node in sel:
            node_type = cmds.nodeType(node)
            if node_type == "joint":
                joints.append(node)
            else:
                controls.append(node)

        # If we got only one type, we can't auto-pair
        if not joints or not controls:
            cmds.warning(
                "Select a mix of joints and controls to auto-pair. "
                "Got {} joints and {} controls.".format(
                    len(joints), len(controls)))
            return

        pair_count = min(len(controls), len(joints))
        if len(controls) != len(joints):
            cmds.warning(
                "Uneven selection: {} controls and {} joints. "
                "Pairing first {}.".format(
                    len(controls), len(joints), pair_count))

        for i in range(pair_count):
            row = self.TBL_fk_mapping.rowCount()
            self.TBL_fk_mapping.insertRow(row)
            # Use short names for display
            ctrl_short = controls[i].split('|')[-1]
            jnt_short = joints[i].split('|')[-1]
            self.TBL_fk_mapping.setItem(
                row, 0, QtWidgets.QTableWidgetItem(ctrl_short))
            self.TBL_fk_mapping.setItem(
                row, 1, QtWidgets.QTableWidgetItem(jnt_short))

    def _fill_active_fk_cell(self, node):
        """Fill the currently selected FK table cell with the given node."""
        current = self.TBL_fk_mapping.currentIndex()
        if not current.isValid():
            cmds.warning("Click a cell in the FK table first.")
            return

        short_name = node.split('|')[-1]
        item = QtWidgets.QTableWidgetItem(short_name)
        self.TBL_fk_mapping.setItem(current.row(), current.column(), item)

    # -----------------------------------------------------------------
    # IK controls list handlers
    # -----------------------------------------------------------------

    def _add_ik_item(self, name, reset=True):
        """Add a checkable item to the IK controls list.

        Args:
            name: Control name.
            reset: If True, the control will be zeroed during FK-to-IK switch.
        """
        item = QtWidgets.QListWidgetItem(name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(
            QtCore.Qt.Checked if reset else QtCore.Qt.Unchecked)
        self.LST_ik_controls.addItem(item)

    def _on_ik_add(self):
        """Add an empty entry to the IK list."""
        self._add_ik_item("", reset=True)

    def _on_ik_remove(self):
        """Remove selected entries from the IK list."""
        for item in reversed(self.LST_ik_controls.selectedItems()):
            row = self.LST_ik_controls.row(item)
            self.LST_ik_controls.takeItem(row)

    def _on_ik_load_selection(self):
        """Load Maya selection into the IK controls list."""
        sel = cmds.ls(selection=True, long=True)
        if not sel:
            cmds.warning("Nothing selected.")
            return

        for node in sel:
            short_name = node.split('|')[-1]
            # Avoid duplicates
            existing = [self.LST_ik_controls.item(i).text()
                        for i in range(self.LST_ik_controls.count())]
            if short_name not in existing:
                self._add_ik_item(short_name, reset=True)

    # -----------------------------------------------------------------
    # Action buttons
    # -----------------------------------------------------------------

    def _get_ui_config(self):
        """Gather configuration from all UI fields."""
        fk_mapping = []
        for row in range(self.TBL_fk_mapping.rowCount()):
            ctrl_item = self.TBL_fk_mapping.item(row, 0)
            jnt_item = self.TBL_fk_mapping.item(row, 1)
            ctrl = ctrl_item.text().strip() if ctrl_item else ""
            jnt = jnt_item.text().strip() if jnt_item else ""
            if ctrl and jnt:
                fk_mapping.append((ctrl, jnt))

        ik_controls = []
        reset_controls = []
        for i in range(self.LST_ik_controls.count()):
            item = self.LST_ik_controls.item(i)
            text = item.text().strip()
            if text:
                ik_controls.append(text)
                if item.checkState() == QtCore.Qt.Checked:
                    reset_controls.append(text)

        blend_attr = self.TXT_blend_attr.text().strip()
        fk_value = self.SPN_fk_value.value()
        ik_value = self.SPN_ik_value.value()
        start_frame = self.SPN_start_frame.value()
        end_frame = self.SPN_end_frame.value()

        sample = 1 if self.OPT_match_keys.isChecked() else self.SPN_step.value()
        use_animLayer = self.CHK_bake_to_layer.isChecked()
        unrollRotations = self.CHK_unroll_rotations.isChecked()

        return {
            'fk_mapping': fk_mapping,
            'ik_controls': ik_controls,
            'reset_controls': reset_controls,
            'blend_attr': blend_attr,
            'fk_value': fk_value,
            'ik_value': ik_value,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'sample': sample,
            'animLayer': use_animLayer,
            'unrollRotations': unrollRotations,
        }

    def _on_switch_to_fk(self):
        """Switch from IK to FK."""
        config = self._get_ui_config()

        if not config['fk_mapping']:
            cmds.warning("No FK mapping defined.")
            return
        if not config['ik_controls']:
            cmds.warning("No IK controls defined.")
            return

        errors = _validate_scene_nodes(
            config['fk_mapping'], config['ik_controls'],
            config['blend_attr'])
        if errors:
            cmds.warning("Validation failed: " + "; ".join(errors))
            return

        switch_to_fk(
            fk_mapping=config['fk_mapping'],
            ik_controls=config['ik_controls'],
            blend_attr=config['blend_attr'],
            fk_value=config['fk_value'],
            start_frame=config['start_frame'],
            end_frame=config['end_frame'],
            sample=config['sample'],
            animLayer=config['animLayer'],
            unrollRotations=config['unrollRotations'])

    def _on_switch_to_ik(self):
        """Switch from FK to IK."""
        config = self._get_ui_config()

        if not config['ik_controls']:
            cmds.warning("No IK controls defined.")
            return

        errors = _validate_scene_nodes(
            [], config['ik_controls'], config['blend_attr'])
        if errors:
            cmds.warning("Validation failed: " + "; ".join(errors))
            return

        switch_to_ik(
            ik_controls=config['ik_controls'],
            blend_attr=config['blend_attr'],
            ik_value=config['ik_value'],
            start_frame=config['start_frame'],
            end_frame=config['end_frame'],
            sample=config['sample'],
            animLayer=config['animLayer'],
            unrollRotations=config['unrollRotations'],
            reset_controls=config['reset_controls'])


# =============================================================================
# Launcher
# =============================================================================

def UI():
    """Launch the IK/FK Switcher UI."""
    IKFKSwitcherUI.run()


def show():
    """Alias for UI()."""
    IKFKSwitcherUI.run()
