import argparse
import os
import importlib
import inspect
from typing import Any, Dict, Set, List, Tuple, Optional, Union, Mapping
from omegaconf import OmegaConf
from trainer import Trainer


def _type_info(val: Any) -> Optional[Tuple[str, str]]:
    if val is None:
        return None
    t = type(val)
    return (t.__module__, t.__name__)


def _list_typename(lst: List[Any]) -> str:
    if not lst:
        return "list[Any]"
    infos = {info for x in lst if (info := _type_info(x)) is not None}
    if len(infos) == 1:
        _, name = next(iter(infos))
        return f"list[{name}]"
    return "list[Any]"


def _collect_attr_types(trainer: Trainer, config: Any) -> Dict[str, str]:
    if not isinstance(config, Dict):
        raise TypeError(f"config must be a Dict bot got {type(config)}")
    types: Dict[str, str] = {}
    for key in config.keys():
        if isinstance(key, str) and key.startswith("_"):
            continue
        val = getattr(trainer, key, None)
        if inspect.isroutine(val):
            continue
        if isinstance(val, list):
            types[key] = _list_typename(val)
        else:
            info = _type_info(val)
            if info is None:
                types[key] = "Any"
            else:
                types[key] = info[1]
    return types


def _collect_imports(trainer: Trainer, config: Any) -> Dict[str, Set[str]]:
    if not isinstance(config, Dict):
        raise TypeError(f"config must be a Dict bot got {type(config)}")
    imports: Dict[str, Set[str]] = {}
    for key in config.keys():
        if isinstance(key, str) and key.startswith("_"):
            continue
        val = getattr(trainer, key, None)
        if inspect.isroutine(val):
            continue
        if isinstance(val, list):
            for x in val:
                info = _type_info(x)
                if info:
                    mod, name = info
                    if mod != "builtins":
                        imports.setdefault(mod, set()).add(name)
        else:
            info = _type_info(val)
            if info:
                mod, name = info
                if mod != "builtins":
                    imports.setdefault(mod, set()).add(name)
    return imports


def _generate_trainer_stub_text(
    trainer: Trainer, types: Dict[str, str], imports: Dict[str, Set[str]]
) -> str:
    lines: List[str] = []
    lines.append("from typing import Any")
    for mod, names in sorted(imports.items()):
        for name in sorted(names):
            lines.append(f"from {mod} import {name}")
    lines.append("")
    lines.append("class Trainer:")
    # cls = type(trainer)
    # method_names: set[str] = set()
    # for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
    #     if name.startswith("_"):
    #         continue
    #     if getattr(func, "__module__", None) != cls.__module__:
    #         continue
    #     method_names.add(name)
    # for name in sorted(method_names):
    #     lines.append(
    #         f"    def {name}(self, *args: Any, **kwargs: Any) -> Any: ..."
    #     )
    for k, v in types.items():
        lines.append(f"    {k}: {v}")
    return "\n".join(lines) + "\n"


def _import_class(qualified_name: str):
    if not isinstance(qualified_name, str) or "." not in qualified_name:
        raise ValueError("class name must be a fully-qualified dotted path")
    module_name, cls_name = qualified_name.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    if not hasattr(mod, cls_name):
        raise ImportError(f"class '{cls_name}' not found in module '{module_name}'")
    cls = getattr(mod, cls_name)
    if not isinstance(cls, type):
        raise TypeError(f"object '{qualified_name}' is not a class")
    return cls


def _generate_subclass_stub_text(
    instance: Trainer, types: Dict[str, str], imports: Dict[str, Set[str]]
) -> str:
    lines: List[str] = []
    lines.append("from typing import Any")
    base_mod = Trainer.__module__
    lines.append(f"from {base_mod} import Trainer")
    for mod, names in sorted(imports.items()):
        for name in sorted(names):
            lines.append(f"from {mod} import {name}")
    lines.append("")
    cls = type(instance)
    subclass_name = cls.__name__
    lines.append(f"class {subclass_name}(Trainer):")
    method_names: Set[str] = set()
    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(func, "__module__", None) != cls.__module__:
            continue
        base_func = getattr(Trainer, name, None)
        if base_func is not None and base_func is func:
            continue
        method_names.add(name)
    for name in sorted(method_names):
        lines.append(
            f"    def {name}(self, *args: Any, **kwargs: Any) -> Any: ..."
        )
    for k, v in types.items():
        lines.append(f"    {k}: {v}")
    return "\n".join(lines) + "\n"


def generate_trainer_base_stub(configs: List[Union[str, Dict[str, Any], Mapping[str, Any]]]) -> str:
    """Generate a Trainer.pyi stub file based on a config-driven Trainer."""
    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("at least one config must be provided")
    merged_conf = None
    for cfg in configs:
        if isinstance(cfg, str):
            if os.path.exists(cfg):
                piece = OmegaConf.load(cfg)
            else:
                piece = OmegaConf.create(cfg)
        elif isinstance(cfg, Mapping):
            piece = OmegaConf.create(cfg)
        else:
            raise TypeError("config must be a mapping or a path string")
        merged_conf = piece if merged_conf is None else OmegaConf.merge(merged_conf, piece)
    config = OmegaConf.to_container(merged_conf, resolve=False)
    trainer = Trainer()
    trainer.configure(config)
    types = _collect_attr_types(trainer, config)
    imports = _collect_imports(trainer, config)

    src_file = inspect.getsourcefile(Trainer) or inspect.getfile(Trainer)
    if not src_file:
        raise RuntimeError("cannot determine source file for Trainer")
    with open(src_file, "r", encoding="utf-8") as f:
        src_text = f.read()

    extra_import_lines: List[str] = []
    for mod, names in sorted(imports.items()):
        for name in sorted(names):
            extra_import_lines.append(f"from {mod} import {name}")

    if extra_import_lines:
        lines = src_text.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            insert_at = i
            break
        lines[insert_at:insert_at] = extra_import_lines
        src_text = "\n".join(lines)
        if not src_text.endswith("\n"):
            src_text += "\n"

    if types:
        attr_lines = [f"    {k}: {v}" for k, v in sorted(types.items())]
        marker = "class Trainer"
        idx = src_text.find(marker)
        if idx == -1:
            raise RuntimeError("class 'Trainer' not found in source")
        line_end = src_text.find("\n", idx)
        if line_end == -1:
            line_end = len(src_text)
        insert_pos = line_end + 1
        src_text = src_text[:insert_pos] + "\n".join(attr_lines) + "\n" + src_text[insert_pos:]

    workspace_root = os.getcwd()
    trainer_mod_name = Trainer.__module__
    top_pkg = trainer_mod_name.split(".", 1)[0]
    stub_root = os.path.join(workspace_root, "typings", top_pkg)
    os.makedirs(stub_root, exist_ok=True)
    module_base = os.path.splitext(os.path.basename(src_file))[0]
    init_pyi = os.path.join(stub_root, "__init__.pyi")
    with open(init_pyi, "w", encoding="utf-8") as f:
        f.write(f"from .{module_base} import Trainer\n__all__ = ['Trainer']\n")
    out_path = os.path.join(stub_root, f"{module_base}.pyi")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(src_text)

    inst_src_path = os.path.join(os.path.dirname(src_file), "instantiate.py")
    if os.path.exists(inst_src_path):
        with open(inst_src_path, "r", encoding="utf-8") as f:
            inst_src = f.read()
        inst_out_path = os.path.join(stub_root, "instantiate.pyi")
        with open(inst_out_path, "w", encoding="utf-8") as f:
            f.write(inst_src)

    _ensure_workspace_stub_path_setting(workspace_root, "./typings")
    _ensure_gitignore_exclude(workspace_root, "typings/")
    return out_path


def generate_subclass_stub(
    configs: List[Union[str, Dict[str, Any], Mapping[str, Any]]], qualified_class: str
) -> str:
    """Generate a .pyi stub for a Trainer subclass using a sample config."""
    cls = _import_class(qualified_class)
    if not issubclass(cls, Trainer):
        expected = f"{Trainer.__module__}.{Trainer.__qualname__}"
        raise TypeError(
            f"class '{qualified_class}' must inherit from {expected}"
        )
    instance = cls()
    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("at least one config must be provided")
    merged_conf = None
    for cfg in configs:
        if isinstance(cfg, str):
            if os.path.exists(cfg):
                piece = OmegaConf.load(cfg)
            else:
                piece = OmegaConf.create(cfg)
        elif isinstance(cfg, Mapping):
            piece = OmegaConf.create(cfg)
        else:
            raise TypeError("config must be a mapping or a path string")
        merged_conf = piece if merged_conf is None else OmegaConf.merge(merged_conf, piece)
    config = OmegaConf.to_container(merged_conf, resolve=False)
    instance.configure(config)
    types = _collect_attr_types(instance, config)
    imports = _collect_imports(instance, config)
    text = _generate_subclass_stub_text(instance, types, imports)
    src_file = inspect.getsourcefile(cls) or inspect.getfile(cls)
    if not src_file:
        raise RuntimeError(
            f"cannot determine source file for class '{qualified_class}'"
        )
    module_dir = os.path.dirname(src_file)
    module_base = os.path.splitext(os.path.basename(src_file))[0]
    out_path = os.path.join(module_dir, f"{module_base}.pyi")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    _ensure_gitignore_exclude(module_dir, os.path.basename(out_path))
    # Also emit to workspace typings mirror for VSCode stubPath resolution
    workspace_root = os.getcwd()
    module_name = qualified_class.rsplit(".", 1)[0]
    typings_dir = os.path.join(workspace_root, "typings", *module_name.split("."))
    os.makedirs(typings_dir, exist_ok=True)
    typings_out = os.path.join(typings_dir, f"{module_base}.pyi")
    try:
        with open(typings_out, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass
    _ensure_workspace_stub_path_setting(workspace_root, "./typings")
    _ensure_gitignore_exclude(workspace_root, "typings/")
    return out_path


def _ensure_gitignore_exclude(dir_path: str, filename: str):
    gi_path = os.path.join(dir_path, ".gitignore")
    line = filename
    exists = os.path.exists(gi_path)
    lines = []
    if exists:
        try:
            with open(gi_path, "r", encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f.readlines()]
        except Exception:
            lines = []
    if line not in lines:
        try:
            with open(gi_path, "a", encoding="utf-8") as f:
                if exists and len(lines) > 0 and not lines[-1].endswith("\n"):
                    f.write("\n")
                f.write(line + "\n")
        except Exception:
            pass


def _ensure_workspace_stub_path_setting(workspace_root: str, stub_root_rel: str):
    vscode_dir = os.path.join(workspace_root, ".vscode")
    os.makedirs(vscode_dir, exist_ok=True)
    settings_path = os.path.join(vscode_dir, "settings.json")
    settings = {}
    if os.path.exists(settings_path):
        try:
            import json

            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            settings = {}
    stub_paths = settings.get("python.analysis.stubPath")
    if isinstance(stub_paths, list):
        if stub_root_rel not in stub_paths:
            stub_paths.append(stub_root_rel)
            settings["python.analysis.stubPath"] = stub_paths
    elif isinstance(stub_paths, str):
        if stub_paths != stub_root_rel:
            settings["python.analysis.stubPath"] = [stub_paths, stub_root_rel]
    else:
        settings["python.analysis.stubPath"] = [stub_root_rel]
    extra_paths = settings.get("python.analysis.extraPaths")
    if isinstance(extra_paths, list):
        if "typings" not in extra_paths:
            extra_paths.append("typings")
            settings["python.analysis.extraPaths"] = extra_paths
    elif isinstance(extra_paths, str):
        if extra_paths != "typings":
            settings["python.analysis.extraPaths"] = [extra_paths, "typings"]
    else:
        settings["python.analysis.extraPaths"] = ["typings"]
    try:
        import json

        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_gen = sub.add_parser("gen_types")
    p_gen.add_argument(
        "--config",
        "-c",
        required=True,
        nargs="+",
        help="One or more configs: YAML/JSON file paths or inline mapping strings",
    )
    p_gen.add_argument(
        "--class",
        "-C",
        dest="qualified_class",
        help="Optional: Fully-qualified Trainer subclass, if generating subclass .pyi",
    )
    args = parser.parse_args()
    if args.cmd == "gen_types":
        if args.qualified_class:
            out = generate_subclass_stub(args.config, args.qualified_class)
        else:
            out = generate_trainer_base_stub(args.config)
        print(out)


if __name__ == "__main__":
    main()
