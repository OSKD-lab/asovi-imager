"""Auto-generated dearpygui editor for PipelineConfig, driven by field_specs.

``build()`` renders one collapsing header per section (DB fields first),
``collect()`` reads the widgets back into a PipelineConfig kwargs dict,
``to_config()`` validates by constructing a ``PipelineConfig`` (reusing all of
its ``__post_init__`` cross-field rules), and ``load()`` repopulates the
widgets from an existing config.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from asvimg import PipelineConfig, load_config, save_config
from asvimg.config import _DB_FIELDS

from .field_specs import FIELD_SPECS, FieldSpec, specs_by_section

_NULLABLE_TEXT = {"output_dir", "exp_name", "template"}


def _tag(name: str) -> str:
    return f"cfgw_{name}"


def _parse_int_list(text: str) -> list[int] | None:
    text = text.strip()
    if not text:
        return None
    return [int(x) for x in text.replace(" ", "").split(",") if x != ""]


def _parse_str_list(text: str) -> list[str] | None:
    items = [x.strip() for x in text.split(",") if x.strip()]
    return items or None


def _clamp(spec: FieldSpec, cast) -> dict[str, Any]:
    """The dpg kwargs that make a FieldSpec's declared min/max actually bind.

    Without these the ranges are decorative: usfac=0 was accepted by the form and
    the config, and died inside the registration kernel with ZeroDivisionError.
    """
    kw: dict[str, Any] = {}
    if spec.min is not None:
        kw["min_value"], kw["min_clamped"] = cast(spec.min), True
    if spec.max is not None:
        kw["max_value"], kw["max_clamped"] = cast(spec.max), True
    return kw


class ConfigForm:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._specs: dict[str, FieldSpec] = {s.name: s for s in FIELD_SPECS}
        # An inline ([src],[ref]) annotation maps to the "cache" combo entry;
        # remember it so an untouched form round-trips the coordinates instead
        # of silently downgrading them to the literal "cache".
        self._loaded_annotation = self.config.annotation
        self._loaded_ica_exclusion = self.config.ica_exclusion

    # --- build ------------------------------------------------------------

    def build(self, parent: str | int, section_footer=None) -> None:
        """Build the form. ``section_footer(section)`` (optional) is called at the
        end of each section — inside its collapsing header — to append extra
        widgets (e.g. an action button)."""
        import dearpygui.dearpygui as dpg

        data = asdict(self.config)
        for section, specs in specs_by_section().items():
            if not specs:
                continue
            header = f"{section}" + ("  (reproducibility)" if section == "Data" else "")
            with dpg.collapsing_header(label=header, parent=parent, default_open=section in ("Data", "Channels")):
                current_group = None
                for spec in specs:
                    # sub-category separator + label (e.g. Frames / Movie / ROI)
                    if spec.group and spec.group != current_group:
                        if current_group is not None:
                            dpg.add_separator()
                        dpg.add_text(f"- {spec.group} -", color=(150, 180, 210))
                        current_group = spec.group
                    self._build_field(dpg, spec, data.get(spec.name))
                if section_footer is not None:
                    section_footer(section)

    def _build_field(self, dpg, spec: FieldSpec, value: Any) -> None:
        tag = _tag(spec.name)
        label = (spec.label or spec.name) + ("  *" if spec.is_db else "")
        kind = spec.kind
        W = 320

        if kind == "bool":
            dpg.add_checkbox(label=label, default_value=bool(value), tag=tag)
        elif kind == "int":
            dpg.add_input_int(
                label=label, default_value=int(value or 0), tag=tag, width=W, step=0,
                **_clamp(spec, int),
            )
        elif kind == "float":
            # add_input_float is backed by a C float: 0.35 comes back as
            # 0.3499999940395355, and every Run writes that back over the user's
            # ops.yaml. add_input_double keeps the value they typed.
            dpg.add_input_double(
                label=label, default_value=float(value or 0.0), tag=tag, width=W, step=0,
                **_clamp(spec, float),
            )
        elif kind == "combo":
            dpg.add_combo(spec.choices, label=label, default_value=self._combo_default(spec, value), tag=tag, width=W)
        elif kind == "text":
            dpg.add_input_text(label=label, default_value="" if value is None else str(value), tag=tag, width=W)
        elif kind == "text_dir":
            with dpg.group(horizontal=True):
                dpg.add_input_text(default_value="" if value is None else str(value), tag=tag, width=W - 30)
                dpg.add_button(label="...", width=26, user_data=tag, callback=self._cb_browse_dir)
                dpg.add_text(label)
        elif kind == "str_list":
            shown = ",".join(value) if isinstance(value, list) else ""
            dpg.add_input_text(label=label, default_value=shown, tag=tag, width=W)
        elif kind == "int_list_or_none":
            shown = ",".join(map(str, value)) if isinstance(value, list) else ""
            dpg.add_input_text(label=label, default_value=shown, tag=tag, width=W, hint="e.g. 1,1,1")
        elif kind == "int_or_none":
            shown = "" if value is None else str(value)
            dpg.add_input_text(label=label, default_value=shown, tag=tag, width=W, hint="(blank = auto/all)")
        elif kind == "annotation_mode":
            dpg.add_combo(spec.choices, label=label, default_value=self._annotation_default(value), tag=tag, width=W)
        elif kind == "ica_mode":
            dpg.add_combo(spec.choices, label=label, default_value=self._ica_mode_default(value), tag=tag, width=W)
        elif kind == "float_pair":
            v0, v1 = (value if isinstance(value, (tuple, list)) and len(value) == 2 else (0.0, 0.0))
            with dpg.group(horizontal=True):
                dpg.add_input_double(label="", default_value=float(v0), tag=tag, width=W // 2 - 10, step=0)
                dpg.add_input_double(label=label, default_value=float(v1), tag=tag + "_b", width=W // 2 - 10, step=0)
        else:
            dpg.add_input_text(label=label, default_value=str(value), tag=tag, width=W)

        if spec.tooltip and dpg.does_item_exist(tag):
            with dpg.tooltip(tag):
                dpg.add_text(spec.tooltip, wrap=400)

    def _cb_browse_dir(self, sender, app_data, user_data) -> None:
        """Folder-picker for a text_dir field; writes the chosen path back."""
        import dearpygui.dearpygui as dpg

        from . import native_dialog

        current = (dpg.get_value(user_data) or "").strip()
        path = native_dialog.open_folder(
            title="Select directory", default_dir=current or None
        )
        if path:
            dpg.set_value(user_data, path)

    @staticmethod
    def _combo_default(spec: FieldSpec, value: Any) -> str:
        if spec.name == "save_roi_signals":
            return "None" if value in (None, False) else str(value)
        return str(value) if value is not None else (spec.choices[0] if spec.choices else "")

    @staticmethod
    def _ica_mode_default(value: Any) -> str:
        # An inline {group: [ics]} dict cannot be shown in a combo; it maps to
        # "cache", and _convert preserves the dict when the user did not touch it.
        if value is False:
            return "False"
        if value in ("cache", "gui"):
            return value
        return "cache"

    @staticmethod
    def _annotation_default(value: Any) -> str:
        if value is False:
            return "False"
        if value in ("cache", "gui"):
            return value
        return "cache"  # tuple/other → show cache (coords editable via YAML)

    # --- collect / validate ----------------------------------------------

    def collect(self) -> dict[str, Any]:
        """Read the widgets back into PipelineConfig kwargs.

        Starts from the loaded config and lets the widgets OVERRIDE it, so a
        field with no widget keeps the value the user wrote in their YAML.
        Collecting only what has a widget would reset such a field to its
        default -- and since every Run writes ops.yaml back out, that default
        would then overwrite the user's file.
        """
        import dearpygui.dearpygui as dpg

        out: dict[str, Any] = asdict(self.config)
        for spec in FIELD_SPECS:
            tag = _tag(spec.name)
            if not dpg.does_item_exist(tag):
                continue
            raw = dpg.get_value(tag)
            out[spec.name] = self._convert(spec, raw, dpg)
        return out

    def _convert(self, spec: FieldSpec, raw: Any, dpg) -> Any:
        kind = spec.kind
        if kind == "bool":
            return bool(raw)
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
        if kind == "combo":
            if spec.name == "save_roi_signals":
                return None if raw == "None" else raw
            return raw
        if kind == "ica_mode":
            if raw == "False":
                return False
            if raw == "cache" and isinstance(self._loaded_ica_exclusion, dict):
                return self._loaded_ica_exclusion  # keep an inline dict from YAML
            return raw
        if kind == "annotation_mode":
            if raw == "False":
                return False
            # "cache" with a remembered coordinate pair → preserve the pair
            # (the combo cannot represent inline coords distinctly).
            if (
                raw == "cache"
                and isinstance(self._loaded_annotation, (list, tuple))
                and len(self._loaded_annotation) == 2
            ):
                return self._loaded_annotation
            return raw
        if kind in ("text", "text_dir"):
            s = str(raw).strip()
            if spec.name in _NULLABLE_TEXT and s == "":
                return None
            return s
        if kind == "str_list":
            return _parse_str_list(str(raw))
        if kind == "int_list_or_none":
            return _parse_int_list(str(raw))
        if kind == "int_or_none":
            s = str(raw).strip()
            return None if s == "" else int(s)
        if kind == "float_pair":
            b = dpg.get_value(_tag(spec.name) + "_b")
            return (float(raw), float(b))
        return raw

    def to_config(self) -> PipelineConfig:
        """Collect + validate. Raises ValueError on invalid config."""
        return PipelineConfig(**self.collect())

    def validate(self) -> tuple[bool, str]:
        try:
            self.to_config()
            return True, ""
        except (ValueError, TypeError) as exc:
            return False, str(exc)

    # --- load / save ------------------------------------------------------

    def load(self, config: PipelineConfig) -> None:
        import dearpygui.dearpygui as dpg

        self.config = config
        self._loaded_annotation = config.annotation
        self._loaded_ica_exclusion = config.ica_exclusion
        data = asdict(config)
        for spec in FIELD_SPECS:
            tag = _tag(spec.name)
            if not dpg.does_item_exist(tag):
                continue
            value = data.get(spec.name)
            self._set_widget(dpg, spec, tag, value)

    def _set_widget(self, dpg, spec: FieldSpec, tag: str, value: Any) -> None:
        kind = spec.kind
        if kind == "bool":
            dpg.set_value(tag, bool(value))
        elif kind == "int":
            dpg.set_value(tag, int(value or 0))
        elif kind == "float":
            dpg.set_value(tag, float(value or 0.0))
        elif kind == "combo":
            dpg.set_value(tag, self._combo_default(spec, value))
        elif kind == "annotation_mode":
            dpg.set_value(tag, self._annotation_default(value))
        elif kind == "ica_mode":
            dpg.set_value(tag, self._ica_mode_default(value))
        elif kind in ("text", "text_dir"):
            dpg.set_value(tag, "" if value is None else str(value))
        elif kind == "str_list":
            dpg.set_value(tag, ",".join(value) if isinstance(value, list) else "")
        elif kind == "int_list_or_none":
            dpg.set_value(tag, ",".join(map(str, value)) if isinstance(value, list) else "")
        elif kind == "int_or_none":
            dpg.set_value(tag, "" if value is None else str(value))
        elif kind == "float_pair":
            v0, v1 = value if isinstance(value, (tuple, list)) and len(value) == 2 else (0.0, 0.0)
            dpg.set_value(tag, float(v0))
            dpg.set_value(tag + "_b", float(v1))
        else:
            dpg.set_value(tag, str(value))

    def load_from_path(self, path: str | Path) -> None:
        self.load(load_config(Path(path)))

    def apply_ops(self, ops_raw: dict) -> None:
        """Apply processing (non-db) fields from a raw dict, leaving the db
        fields (input/output/format/...) untouched.

        Deprecated/renamed keys are handled so older presets / default files
        still apply cleanly."""
        import dearpygui.dearpygui as dpg

        from asvimg.config import _apply_renames, _strip_deprecated

        raw = dict(ops_raw)
        _strip_deprecated(raw)
        _apply_renames(raw)
        ops = {k: v for k, v in raw.items() if k not in _DB_FIELDS}
        try:
            base = self.collect()  # keep current db fields
        except Exception:  # noqa: BLE001
            base = asdict(PipelineConfig())
        cfg = PipelineConfig(**{**base, **ops})
        self.config = cfg
        self._loaded_annotation = cfg.annotation
        self._loaded_ica_exclusion = cfg.ica_exclusion
        data = asdict(cfg)
        for spec in FIELD_SPECS:
            if spec.name in _DB_FIELDS:
                continue
            tag = _tag(spec.name)
            if dpg.does_item_exist(tag):
                self._set_widget(dpg, spec, tag, data.get(spec.name))

    def apply_ops_preset(self, path: str | Path) -> None:
        """Apply the processing (non-db) fields from a preset YAML file."""
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.apply_ops(raw)

    def restore_ops_defaults(self) -> None:
        """Reset processing (non-db) fields to factory ``PipelineConfig()``
        defaults, keeping the db fields."""
        self.apply_ops(asdict(PipelineConfig()))

    def save_ops(self, path: str | Path) -> None:
        """Write the current processing (non-db) fields to a YAML file
        (creating parent dirs); validates the config first."""
        import yaml

        data = asdict(self.to_config())
        ops = {k: v for k, v in data.items() if k not in _DB_FIELDS}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(ops, f, sort_keys=False, allow_unicode=True)

    def save_to(self, output_dir: str | Path) -> None:
        """Validate, then write db.yaml + ops.yaml to ``output_dir``."""
        save_config(self.to_config(), Path(output_dir))
