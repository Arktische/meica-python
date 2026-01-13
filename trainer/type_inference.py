import importlib
import inspect
import logging
from typing import (
    Any,
    Dict,
    List,
    Union,
    Optional,
    get_type_hints,
)
from .instantiate import (
    format_keys_path,
    set_dict_by_keys,
    resolve_string_value,
    check_reserved_prefix,
    TYPE,
    OBJECT,
    CALL,
    METHOD,
    RefTracker,
)

_LOGGER = logging.getLogger("meica")


def _get_object_from_path(path: str) -> Any:
    """More robust version of _get_type that can traverse attributes."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            obj = importlib.import_module(mod_name)
            for part in parts[i:]:
                obj = getattr(obj, part)
            return obj
        except (ImportError, AttributeError, ModuleNotFoundError):
            continue
    raise ImportError(f"Cannot find object at path {path}")


def _resolve_type(obj: Any) -> Any:
    """Infer the return type of a callable or return the class itself."""
    if inspect.isclass(obj):
        return obj

    if callable(obj):
        try:
            # Try to get return type from signature
            sig = inspect.signature(obj)
            ret = sig.return_annotation

            if ret is inspect.Signature.empty:
                # Try get_type_hints for better resolution
                hints = get_type_hints(obj)
                ret = hints.get("return", Any)

            # Handle Self (Python 3.11+) or string 'Self'
            if str(ret) == "typing.Self" or ret == "Self":
                # If it's a method, we can try to find the class it belongs to
                if inspect.ismethod(obj):
                    return (
                        obj.__self__
                        if inspect.isclass(obj.__self__)
                        else type(obj.__self__)
                    )
                elif inspect.isfunction(obj):
                    # Check if it's a classmethod or staticmethod defined in a class
                    qualname = getattr(obj, "__qualname__", "")
                    if "." in qualname:
                        cls_name = qualname.rsplit(".", 1)[0]
                        # This is a bit hacky, but might work if the class is in the same module
                        mod = importlib.import_module(obj.__module__)
                        return getattr(mod, cls_name, Any)

            if ret is Any or ret is inspect.Signature.empty:
                return Any
            return ret
        except Exception as e:
            _LOGGER.debug(f"Failed to infer type for {obj}: {e}")
            return Any

    return type(obj)


def dry_instantiate(
    root: Dict[Any, Any],
    keys: List[Union[str, int]],
    node: Any,
    tracker: Optional[RefTracker] = None,
    allow_object_processing: bool = True,
):
    """Similar to instantiate but only performs type inference without creating real objects."""
    if tracker is None:
        tracker = RefTracker()

    path_str = format_keys_path(keys) if keys else ""

    if tracker.is_visiting(path_str):
        raise ValueError(f"circular reference detected at '{path_str}'")
    if tracker.is_resolved(path_str):
        return
    tracker.mark_visiting(path_str)

    if isinstance(node, dict):
        # Recursively process children first (to resolve references later)
        for key, value in node.items():
            check_reserved_prefix(key)
            dry_instantiate(
                root=root,
                keys=[*keys, key],
                node=value,
                tracker=tracker,
                allow_object_processing=(key != CALL),
            )

        # Infer type for TYPE nodes
        if TYPE in node:
            try:
                obj = _get_object_from_path(node[TYPE])
                res_type = _resolve_type(obj)
                set_dict_by_keys(root, keys, res_type)
            except Exception as e:
                _LOGGER.warning(f"Failed to resolve type {node.get(TYPE)}: {e}")
                set_dict_by_keys(root, keys, Any)

        # Handle OBJECT nodes (keep as is or infer type)
        elif allow_object_processing and OBJECT in node:
            # If it's already an object, we just take its type
            val = node[OBJECT]
            set_dict_by_keys(root, keys, _resolve_type(val))

        # Handle CALL nodes
        if CALL in node:
            call_spec = node[CALL]
            if isinstance(call_spec, dict):
                obj = call_spec.get(OBJECT)
                method_name = call_spec.get(METHOD)
                if obj is not None and method_name is not None:
                    try:
                        # obj here might be a class or a type resolved from a previous step
                        if inspect.isclass(obj):
                            method = getattr(obj, method_name, None)
                        else:
                            method = getattr(obj, method_name, None)

                        if method:
                            res_type = _resolve_type(method)
                            set_dict_by_keys(root, keys, res_type)
                        else:
                            set_dict_by_keys(root, keys, Any)
                    except Exception:
                        set_dict_by_keys(root, keys, Any)
            else:
                set_dict_by_keys(root, keys, Any)

    elif isinstance(node, list):
        for index, value in enumerate(node):
            dry_instantiate(root=root, keys=[*keys, index], node=value, tracker=tracker)

    else:
        if isinstance(node, str):
            # References are still resolved, but they will point to types now
            resolved = resolve_string_value(
                root=root,
                keys=keys,
                value=node,
                tracker=tracker,
                path_str=path_str,
            )
            if resolved is not node:
                set_dict_by_keys(root, keys, resolved)

    tracker.mark_resolved(path_str)
