import importlib
import inspect
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    List,
    Dict,
    Set,
    Optional,
    TypeVar,
    Union,
    Callable,
)

TYPE = "type"
ARGS = "args"
OBJECT = "object"
CALL = "call"
METHOD = "method"


PRESERVED_KEYS = [TYPE, ARGS, OBJECT, CALL, METHOD]

PRESERVED_KEY_PREFIX = "_"


def _format_keys_path(keys: List[Union[str, int]]):
    parts = []
    for p in keys:
        if isinstance(p, str):
            parts.append(p)
        else:
            if not parts:
                parts.append(str(p))
            else:
                parts[-1] = f"{parts[-1]}[{p}]"
    return ".".join(parts)


def _get_type(string: str, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _check_reserved_prefix(key: str):
    if isinstance(key, str) and key.startswith(PRESERVED_KEY_PREFIX):
        raise ValueError(
            f"invalid config key with reserved prefix '{PRESERVED_KEY_PREFIX}': {key}"
        )


def _set_dict_by_keys(root: Dict[Any, Any], keys: List[Union[str, int]], val: Any):
    node = root
    for idx, key in enumerate(keys):
        if isinstance(key, str) or isinstance(key, int):
            if idx == len(keys) - 1:
                node[key] = val
            else:
                node = node[key]
        else:
            raise ValueError(
                f"only str or int key type is supported, but got: {type(key)}"
            )


def _get_dict_by_keys(root: Dict[Any, Any], keys: List[Union[str, int]]) -> Any:
    node = root
    for key in keys:
        if isinstance(key, str) or isinstance(key, int):
            node = node[key]
        else:
            raise ValueError(
                f"only str or int key type is supported, but got: {type(key)}"
            )
    return node


def _get_dict_by_keys_path(root: Dict[Any, Any], keys_path: str):
    keys = _parse_keys_path(keys_path)
    return _get_dict_by_keys(root, keys)


def _set_dict_by_keys_path(root: Dict[Any, Any], keys_path: str, val: Any):
    keys = _parse_keys_path(keys_path)
    _set_dict_by_keys(root, keys, val)


def _parse_keys_path(keys_path: str) -> List[Union[str, int]]:
    parts = []
    for token in keys_path.split("."):
        if "[" in token and token.endswith("]"):
            name, idx_str = token[:-1].split("[", 1)
            parts.append(name)
            parts.append(int(idx_str))
        else:
            if token.isdigit():
                parts.append(int(token))
            else:
                parts.append(token)
    return parts


@dataclass
class RefTracker:
    """Tracks reference resolution state during config instantiation.

    Maintains two sets of path strings:
    - visiting: paths currently being resolved (for cycle detection)
    - resolved: paths already resolved (to skip re-processing)
    """

    visiting: Set[str] = field(default_factory=set)
    resolved: Set[str] = field(default_factory=set)

    def is_visiting(self, path: str) -> bool:
        return path in self.visiting

    def is_resolved(self, path: str) -> bool:
        return path in self.resolved

    def mark_visiting(self, path: str) -> None:
        self.visiting.add(path)

    def mark_resolved(self, path: str) -> None:
        self.visiting.discard(path)
        self.resolved.add(path)


_T = TypeVar("_T")


class NodeRef(Generic[_T]):
    def __init__(self, root: Any, keys: List[Union[str, int]]):
        self._root = root
        self._keys = keys

    @property
    def keys(self) -> List[Union[str, int]]:
        return self._keys

    @property
    def keys_path(self) -> str:
        return _format_keys_path(self._keys)

    @property
    def value(self) -> _T:
        return _get_dict_by_keys(self._root, self._keys)

    @value.setter
    def value(self, val: _T):
        _set_dict_by_keys(self._root, self._keys, val)

    def set(self, val: _T):
        self.value = val


class AttributeRef(NodeRef):
    def __init__(self, obj: Any, attr_name: str):
        super().__init__(root=obj, keys=[attr_name])

    @property
    def value(self) -> Any:
        return getattr(self._root, str(self._keys[0]))

    @value.setter
    def value(self, val: Any):
        setattr(self._root, str(self._keys[0]), val)


class PostConfigureContext:
    def __init__(
        self,
        filters: List[Callable[[NodeRef], bool]],
        transform: Callable[[List[NodeRef]], None],
    ):
        """Collect matching _DictRef nodes and apply a bulk transform at the end."""
        self.filters: List[Callable[[NodeRef], bool]] = filters
        self.candidates: List[List[NodeRef]] = [[] for _ in range(len(self.filters))]
        self.transform: Callable[[List[NodeRef]], None] = transform
        self.value_ids: List[Set[int]] = [set() for _ in range(len(self.filters))]

    def collect(self, node: NodeRef):
        for i, filter in enumerate(self.filters):
            if filter(node):
                value_id = id(node.value)
                if value_id not in self.value_ids[i]:
                    self.value_ids[i].add(value_id)
                    self.candidates[i].append(node)

    def apply(self):
        self.transform(*self.candidates)


# def apply_post_configure(root: Dict[Any, Any], contexts: List[PostConfigureContext]):
#     _apply_post_configure_core(root=root, keys=[], node=root, contexts=contexts)
#     for ctx in contexts:
#         ctx.apply()


def apply_post_configure_to_dict(
    root: Any,
    keys: List[Union[str, int]],
    node: Any,
    contexts: List[PostConfigureContext],
):
    if isinstance(node, dict):
        for key, value in node.items():
            apply_post_configure_to_dict(
                root=root,
                keys=[*keys, key],
                node=value,
                contexts=contexts,
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            apply_post_configure_to_dict(
                root=root,
                keys=[*keys, index],
                node=item,
                contexts=contexts,
            )
    else:
        for ctx in contexts:
            ctx.collect(NodeRef(root, keys))


def apply_post_configure_to_class(
    obj: Any,
    contexts: List[PostConfigureContext],
):
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(obj, attr_name)
        except Exception:
            continue
        # We only collect immediate attributes that match the types we care about
        # In this unified form, we wrap it in AttributeRef
        ref = AttributeRef(obj, attr_name)
        for ctx in contexts:
            ctx.collect(ref)


def _resolve_reference_value(
    root: Dict[Any, Any],
    keys_path: str,
    tracker: RefTracker,
    current_path: str,
):
    target_keys = _parse_keys_path(keys_path)
    target_path_str = _format_keys_path(target_keys)
    if tracker.is_visiting(target_path_str):
        raise ValueError(
            f"circular reference detected at '{current_path}' -> '{target_path_str}'"
        )
    try:
        target_val = _get_dict_by_keys(root, target_keys)
    except KeyError as e:
        raise ValueError(
            f"reference target not found at '{current_path}' -> '{target_path_str}'"
        ) from e
    instantiate(root=root, keys=target_keys, node=target_val, tracker=tracker)
    try:
        resolved_val = _get_dict_by_keys(root, target_keys)
    except KeyError as e:
        raise ValueError(
            f"reference target not found at '{current_path}' -> '{target_path_str}'"
        ) from e
    return resolved_val


def _resolve_string_value(
    root: Dict[Any, Any],
    keys: List[Union[str, int]],
    value: str,
    tracker: RefTracker,
    path_str: str,
):
    m = re.fullmatch(r"\$\{\s*([^\}]+?)\s*\}", value)
    if m:
        keys_path = m.group(1).strip()
        if not keys_path:
            raise ValueError(f"invalid empty reference at '{path_str}'")
        resolved_val = _resolve_reference_value(
            root=root,
            keys_path=keys_path,
            tracker=tracker,
            current_path=path_str,
        )
        return resolved_val

    s = value
    n = len(s)
    parts: List[str] = []
    i = 0
    has_placeholder = False
    changed = False
    while i < n:
        ch = s[i]
        if ch == "$":
            if i + 2 < n and s[i + 1] == "$" and s[i + 2] == "{":
                parts.append("${")
                changed = True
                i += 3
                continue
            if i + 1 < n and s[i + 1] == "{":
                end = s.find("}", i + 2)
                if end == -1:
                    raise ValueError(f"invalid reference syntax at '{path_str}'")
                expr = s[i + 2 : end].strip()
                if not expr:
                    raise ValueError(f"invalid empty reference at '{path_str}'")
                resolved_val = _resolve_reference_value(
                    root=root,
                    keys_path=expr,
                    tracker=tracker,
                    current_path=path_str,
                )
                parts.append(str(resolved_val))
                has_placeholder = True
                changed = True
                i = end + 1
                continue
        parts.append(ch)
        i += 1

    if not changed:
        return value
    return "".join(parts)


def instantiate(
    root: Dict[Any, Any],
    keys: List[Union[str, int]],
    node: Any,
    tracker: Optional[RefTracker] = None,
    allow_object_processing: bool = True,
):
    """Instantiate a configuration tree with type construction and references.

    - Recursively walks dictionaries, lists, and strings.
    - Supports reference strings `${a.b.c[0].e}` that point to other paths in the same config.
    - When encountering a dict with a `type` key, constructs an object via `args` if provided, otherwise returns the type itself.

    Parameters:
        root: The root configuration dictionary to mutate in-place.
        keys: The current traversal path represented as a list of strings or integers.
        val: The current value at `keys` to process.
        tracker: Reference tracking state object for cycle detection and memoization.
    """
    # Initialize tracker if not provided
    if tracker is None:
        tracker = RefTracker()

    path_str = _format_keys_path(keys) if keys else ""

    # Cycle detection and memoization
    if tracker.is_visiting(path_str):
        raise ValueError(f"circular reference detected at '{path_str}'")
    if tracker.is_resolved(path_str):
        return
    tracker.mark_visiting(path_str)

    # Walk mapping values
    if isinstance(node, dict):
        for key, value in node.items():
            _check_reserved_prefix(key)
            instantiate(
                root=root,
                keys=[*keys, key],
                node=value,
                tracker=tracker,
                allow_object_processing=(key != CALL),
            )

        # Construct instance or return type constant
        if TYPE in node:
            type_obj = _get_type(node[TYPE])
            node_keys = [*node.keys()]
            node_keys.remove(TYPE)
            if ARGS in node:
                node_keys.remove(ARGS)
                if not inspect.isclass(type_obj) and not callable(type_obj):
                    raise ValueError(f"Invalid type: {type_obj}")
                args = node[ARGS]
                if isinstance(args, dict):
                    obj = type_obj(**args)
                elif isinstance(args, list):
                    obj = type_obj(*args)
                else:
                    raise ValueError(f"Invalid args type: {type(args)}")
            else:
                obj = type_obj

            for attr_key in node_keys:
                attr_args = node[attr_key]
                if hasattr(obj, attr_key):
                    attr = getattr(obj, attr_key)
                    if callable(attr):
                        if isinstance(attr_args, dict):
                            attr(**attr_args)
                        elif isinstance(attr_args, list):
                            attr(*attr_args)
                        else:
                            attr()
                    else:
                        if attr_args is not None:
                            setattr(obj, attr_key, attr_args)
                        else:
                            pass
                else:
                    setattr(obj, attr_key, attr_args)
            _set_dict_by_keys(root, keys, obj)

        if allow_object_processing and OBJECT in node:
            obj = node[OBJECT]
            node_keys = [*node.keys()]
            node_keys.remove(OBJECT)

            for attr_key in node_keys:
                attr_args = node[attr_key]
                if hasattr(obj, attr_key):
                    attr = getattr(obj, attr_key)
                    if callable(attr):
                        if isinstance(attr_args, dict):
                            attr(**attr_args)
                        elif isinstance(attr_args, list):
                            attr(*attr_args)
                        else:
                            attr()
                    else:
                        if attr_args is not None:
                            setattr(obj, attr_key, attr_args)
                        else:
                            pass

                else:
                    setattr(obj, attr_key, attr_args)

        if CALL in node:
            call_spec = node[CALL]
            if not isinstance(call_spec, dict):
                raise ValueError(f"Value of '{CALL}' must be a dict")

            obj = call_spec.get(OBJECT)
            method_name = call_spec.get(METHOD)
            args = call_spec.get(ARGS)

            if obj is None:
                raise ValueError(f"'{CALL}' spec must contain '{OBJECT}'")
            if method_name is None:
                raise ValueError(f"'{CALL}' spec must contain '{METHOD}'")

            if not hasattr(obj, method_name):
                raise ValueError(f"Object {obj} has no attribute '{method_name}'")

            method = getattr(obj, method_name)
            if not callable(method):
                raise ValueError(f"Attribute '{method_name}' of {obj} is not callable")

            if args is None:
                result = method()
            elif isinstance(args, list):
                result = method(*args)
            elif isinstance(args, dict):
                result = method(**args)
            else:
                raise ValueError(f"'{ARGS}' in '{CALL}' must be list or dict")

            _set_dict_by_keys(root, keys, result)

    # Walk list values
    elif isinstance(node, list):
        for index, value in enumerate(node):
            instantiate(root=root, keys=[*keys, index], node=value, tracker=tracker)

    # Resolve reference strings and write back
    else:
        if isinstance(node, str):
            resolved = _resolve_string_value(
                root=root,
                keys=keys,
                value=node,
                tracker=tracker,
                path_str=path_str,
            )
            if resolved is not node:
                _set_dict_by_keys(root, keys, resolved)

    # Mark path as resolved
    tracker.mark_resolved(path_str)
