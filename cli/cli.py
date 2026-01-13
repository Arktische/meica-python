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
    if isinstance(val, type):
        t = val
    else:
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
        lines.append(f"    def {name}(self, *args: Any, **kwargs: Any) -> Any: ...")
    for k, v in types.items():
        lines.append(f"    {k}: {v}")
    return "\n".join(lines) + "\n"


def _load_config(configs: List[Union[str, Dict[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
    merged_conf = None
    for cfg in configs:
        if isinstance(cfg, str):
            piece = OmegaConf.load(cfg) if os.path.exists(cfg) else OmegaConf.create(cfg)
        elif isinstance(cfg, Mapping):
            piece = OmegaConf.create(dict(cfg))
        else:
            raise TypeError("config must be a mapping or a path string")
        merged_conf = piece if merged_conf is None else OmegaConf.merge(merged_conf, piece)
    container = OmegaConf.to_container(merged_conf, resolve=False)
    if not isinstance(container, dict):
        raise TypeError(f"merged config must be a dict, got {type(container)}")
    return container  # type: ignore


def _patch_trainer_source(trainer: Trainer, config: Dict[str, Any]) -> str:
    types = _collect_attr_types(trainer, config)
    imports = _collect_imports(trainer, config)

    src_file = inspect.getsourcefile(type(trainer)) or inspect.getfile(type(trainer))
    if not src_file:
        raise RuntimeError(f"cannot determine source file for {type(trainer)}")
    with open(src_file, "r", encoding="utf-8") as f:
        src_text = f.read()

    extra_imports = [f"from {mod} import {name}" for mod, names in sorted(imports.items()) for name in sorted(names)]
    if extra_imports:
        lines = src_text.splitlines()
        insert_at = next((i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("#")), 0)
        lines[insert_at:insert_at] = extra_imports
        src_text = "\n".join(lines) + "\n"

    if types:
        attr_lines = [f"    {k}: {v}" for k, v in sorted(types.items())]
        marker = f"class {type(trainer).__name__}"
        idx = src_text.find(marker)
        if idx != -1:
            line_end = src_text.find("\n", idx)
            insert_pos = (line_end + 1) if line_end != -1 else len(src_text)
            src_text = src_text[:insert_pos] + "\n".join(attr_lines) + "\n" + src_text[insert_pos:]
    return src_text


def _ensure_workspace_settings(workspace_root: str):
    _ensure_workspace_stub_path_setting(workspace_root, "./typings")
    _ensure_gitignore_exclude(workspace_root, "typings/")


def generate_trainer_base_stub(
    configs: List[Union[str, Dict[str, Any], Mapping[str, Any]]],
) -> str:
    """Generate .pyi stub files for the entire trainer package."""
    config = _load_config(configs)
    trainer = Trainer()
    trainer.configure(True, config)

    trainer_cls_file = inspect.getsourcefile(Trainer) or inspect.getfile(Trainer)
    package_dir = os.path.dirname(trainer_cls_file)
    package_name = Trainer.__module__.split(".")[0]
    workspace_root = os.getcwd()
    stub_root = os.path.join(workspace_root, "typings", package_name)
    os.makedirs(stub_root, exist_ok=True)

    main_out_path = ""
    for filename in os.listdir(package_dir):
        if not filename.endswith(".py"):
            continue
        src_path = os.path.join(package_dir, filename)
        module_base = os.path.splitext(filename)[0]
        out_path = os.path.join(stub_root, f"{module_base}.pyi")

        if src_path == trainer_cls_file:
            content = _patch_trainer_source(trainer, config)
            main_out_path = out_path
        else:
            with open(src_path, "r", encoding="utf-8") as f:
                content = f.read()

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Ensure __init__.pyi exports Trainer
    init_pyi = os.path.join(stub_root, "__init__.pyi")
    if not os.path.exists(init_pyi):
        module_base = os.path.splitext(os.path.basename(trainer_cls_file))[0]
        with open(init_pyi, "w", encoding="utf-8") as f:
            f.write(f"from .{module_base} import Trainer\n__all__ = ['Trainer']\n")

    _ensure_workspace_settings(workspace_root)
    return main_out_path


def generate_subclass_stub(
    configs: List[Union[str, Dict[str, Any], Mapping[str, Any]]], qualified_class: str
) -> str:
    """Generate a .pyi stub for a Trainer subclass using a sample config."""
    cls = _import_class(qualified_class)
    if not issubclass(cls, Trainer):
        expected = f"{Trainer.__module__}.{Trainer.__qualname__}"
        raise TypeError(f"class '{qualified_class}' must inherit from {expected}")

    config = _load_config(configs)
    instance = cls()
    instance.configure(True, config)

    text = _generate_subclass_stub_text(instance, _collect_attr_types(instance, config), _collect_imports(instance, config))

    src_file = inspect.getsourcefile(cls) or inspect.getfile(cls)
    if not src_file:
        raise RuntimeError(f"cannot determine source file for class '{qualified_class}'")

    module_dir = os.path.dirname(src_file)
    module_base = os.path.splitext(os.path.basename(src_file))[0]
    out_path = os.path.join(module_dir, f"{module_base}.pyi")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    _ensure_gitignore_exclude(module_dir, os.path.basename(out_path))

    # Also emit to workspace typings mirror
    workspace_root = os.getcwd()
    module_parts = qualified_class.rsplit(".", 1)[0].split(".")
    typings_dir = os.path.join(workspace_root, "typings", *module_parts)
    os.makedirs(typings_dir, exist_ok=True)
    try:
        with open(os.path.join(typings_dir, f"{module_base}.pyi"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

    _ensure_workspace_settings(workspace_root)
    return out_path




def _ensure_gitignore_exclude(dir_path: str, filename: str):
    gi_path = os.path.join(dir_path, ".gitignore")
    try:
        lines = open(gi_path).read().splitlines() if os.path.exists(gi_path) else []
        if filename not in lines:
            with open(gi_path, "a") as f:
                if lines and not lines[-1].endswith("\n"): f.write("\n")
                f.write(f"{filename}\n")
    except Exception: pass


def _ensure_workspace_stub_path_setting(workspace_root: str, stub_root_rel: str):
    vscode_dir = os.path.join(workspace_root, ".vscode")
    os.makedirs(vscode_dir, exist_ok=True)
    settings_path = os.path.join(vscode_dir, "settings.json")
    try:
        import json
        settings = json.load(open(settings_path)) if os.path.exists(settings_path) else {}
        
        def update_list(key, val):
            curr = settings.get(key, [])
            if isinstance(curr, str): curr = [curr]
            if val not in curr: settings[key] = curr + [val]

        update_list("python.analysis.stubPath", stub_root_rel)
        update_list("python.analysis.extraPaths", "typings")
        json.dump(settings, open(settings_path, "w"), indent=2)
    except Exception: pass


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
