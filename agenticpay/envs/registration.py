"""Environment Registration System

Reference Gymnasium's design, provides environment registration and creation functionality.
"""

from __future__ import annotations

import contextlib
import copy
import difflib
import importlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable

from agenticpay.core import BaseEnv


# Environment ID regular expression: [namespace/](env_name)[-v(version)]
ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)

__all__ = [
    "registry",
    "current_namespace",
    "EnvSpec",
    "register",
    "make",
    "spec",
    "pprint_registry",
    "namespace",
]


class EnvCreator(Protocol):
    """Type protocol for environment creation functions"""

    def __call__(self, **kwargs: Any) -> BaseEnv: ...


@dataclass
class EnvSpec:
    """Environment specification class
    
    Used to store complete configuration information of an environment, including entry point, parameters, etc.
    
    Attributes:
        id: Environment ID, used to create environment
        entry_point: Environment entry point, can be a class or string path
        max_episode_steps: Maximum step limit
        kwargs: Parameter dictionary passed to environment constructor
    """

    id: str
    entry_point: EnvCreator | str | None = field(default=None)
    max_episode_steps: int | None = field(default=None)
    kwargs: dict = field(default_factory=dict)

    # post-init attributes
    namespace: str | None = field(init=False)
    name: str = field(init=False)
    version: int | None = field(init=False)

    def __post_init__(self):
        """Parse environment ID, extract namespace, name and version"""
        self.namespace, self.name, self.version = parse_env_id(self.id)

    def make(self, **kwargs: Any) -> BaseEnv:
        """Create environment instance using this specification"""
        return make(self, **kwargs)


# Global environment registry
registry: dict[str, EnvSpec] = {}
current_namespace: str | None = None


def parse_env_id(env_id: str) -> tuple[str | None, str, int | None]:
    """Parse environment ID string format
    
    Args:
        env_id: Environment ID, format is [namespace/](env_name)[-v(version)]
        
    Returns:
        (namespace, name, version) tuple
        
    Raises:
        ValueError: If environment ID format is invalid
    """
    match = ENV_ID_RE.fullmatch(env_id)
    if not match:
        raise ValueError(
            f"Invalid environment ID format: {env_id}. "
            f"Format should be: [namespace/](env_name)[-v(version)]"
        )
    ns, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)
    return ns, name, version


def get_env_id(ns: str | None, name: str, version: int | None) -> str:
    """Generate complete environment ID from namespace, name and version
    
    Args:
        ns: Namespace
        name: Environment name
        version: Version number
        
    Returns:
        Complete environment ID
    """
    full_name = name
    if ns is not None:
        full_name = f"{ns}/{name}"
    if version is not None:
        full_name = f"{full_name}-v{version}"
    return full_name


def find_highest_version(ns: str | None, name: str) -> int | None:
    """Find the highest version of specified environment
    
    Args:
        ns: Namespace
        name: Environment name
        
    Returns:
        Highest version number, returns None if not exists
    """
    versions = [
        env_spec.version
        for env_spec in registry.values()
        if env_spec.namespace == ns
        and env_spec.name == name
        and env_spec.version is not None
    ]
    return max(versions, default=None)


def _check_namespace_exists(ns: str | None):
    """Check if namespace exists"""
    if ns is None:
        return
    
    namespaces = {
        env_spec.namespace
        for env_spec in registry.values()
        if env_spec.namespace is not None
    }
    if ns in namespaces:
        return
    
    suggestion = (
        difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    )
    if suggestion:
        suggestion_msg = f"Did you mean: `{suggestion[0]}`?"
    else:
        suggestion_msg = f"Have you installed a package containing namespace {ns}?"
    
    raise ValueError(f"Namespace {ns} does not exist. {suggestion_msg}")


def _check_name_exists(ns: str | None, name: str):
    """Check if environment name exists"""
    _check_namespace_exists(ns)
    
    names = {
        env_spec.name for env_spec in registry.values() if env_spec.namespace == ns
    }
    if name in names:
        return
    
    suggestion = difflib.get_close_matches(name, names, n=1)
    namespace_msg = f" in namespace {ns}" if ns else ""
    suggestion_msg = f" Did you mean: `{suggestion[0]}`?" if suggestion else ""
    
    raise ValueError(
        f"Environment `{name}` does not exist{namespace_msg}.{suggestion_msg}"
    )


def _check_version_exists(ns: str | None, name: str, version: int | None):
    """Check if environment version exists"""
    if get_env_id(ns, name, version) in registry:
        return
    
    _check_name_exists(ns, name)
    if version is None:
        return
    
    message = f"Environment version `v{version}` for environment `{get_env_id(ns, name, None)}` does not exist."
    
    env_specs = [
        env_spec
        for env_spec in registry.values()
        if env_spec.namespace == ns and env_spec.name == name
    ]
    env_specs = sorted(env_specs, key=lambda env_spec: int(env_spec.version or -1))
    
    versioned_specs = [
        env_spec for env_spec in env_specs if env_spec.version is not None
    ]
    
    latest_spec = max(versioned_specs, key=lambda env_spec: env_spec.version, default=None)  # type: ignore
    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{env_spec.version}`" for env_spec in env_specs)
        message += f" Available versions: [ {version_list_msg} ]."
        raise ValueError(message)
    
    if latest_spec is not None and version < latest_spec.version:
        raise ValueError(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. "
            f"Please use `{latest_spec.id}` instead."
        )


def _check_spec_register(testing_spec: EnvSpec):
    """Check if specification can be registered"""
    latest_versioned_spec = max(
        (
            env_spec
            for env_spec in registry.values()
            if env_spec.namespace == testing_spec.namespace
            and env_spec.name == testing_spec.name
            and env_spec.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )
    
    unversioned_spec = next(
        (
            env_spec
            for env_spec in registry.values()
            if env_spec.namespace == testing_spec.namespace
            and env_spec.name == testing_spec.name
            and env_spec.version is None
        ),
        None,
    )
    
    if unversioned_spec is not None and testing_spec.version is not None:
        raise ValueError(
            f"Cannot register versioned environment `{testing_spec.id}`, because unversioned environment "
            f"`{unversioned_spec.id}` already exists."
        )
    elif latest_versioned_spec is not None and testing_spec.version is None:
        raise ValueError(
            f"Cannot register unversioned environment `{testing_spec.id}`, because versioned environment "
            f"`{latest_versioned_spec.id}` already exists."
        )


def load_env_creator(name: str) -> EnvCreator:
    """Load environment creation function
    
    Args:
        name: Environment name, format is "module:ClassName"
        
    Returns:
        Environment creation function
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


@contextlib.contextmanager
def namespace(ns: str):
    """Namespace context manager
    
    Used to temporarily set the current namespace.
    
    Args:
        ns: Namespace name
        
    Example:
        >>> with namespace("my_namespace"):
        ...     register(id="MyEnv-v0", entry_point=MyEnv)
        ...     # The actual registered ID is "my_namespace/MyEnv-v0"
    """
    global current_namespace
    old_namespace = current_namespace
    current_namespace = ns
    try:
        yield
    finally:
        current_namespace = old_namespace


def register(
    id: str,
    entry_point: EnvCreator | str | None = None,
    max_episode_steps: int | None = None,
    kwargs: dict | None = None,
):
    """Register environment
    
    Register an environment in the global registry so it can be created via the `make()` function.
    
    Args:
        id: Environment ID, format is [namespace/](env_name)[-v(version)]
        entry_point: Environment entry point, can be a class object or string path (e.g., "module:ClassName")
        max_episode_steps: Maximum step limit
        kwargs: Parameter dictionary passed to environment constructor
        
    Example:
        >>> register(
        ...     id="my_env/Negotiation-v0",
        ...     entry_point=NegotiationEnv,
        ...     max_episode_steps=100,
        ...     kwargs={"default_param": "value"}
        ... )
    """
    if entry_point is None:
        raise ValueError("`entry_point` must be provided")
    
    ns, name, version = parse_env_id(id)
    
    if kwargs is None:
        kwargs = dict()
    
    if current_namespace is not None:
        if kwargs.get("namespace") is not None and kwargs.get("namespace") != current_namespace:
            import warnings
            warnings.warn(
                f"Custom namespace `{kwargs.get('namespace')}` will be overridden by namespace `{current_namespace}`."
            )
        ns_id = current_namespace
    else:
        ns_id = ns
    
    full_env_id = get_env_id(ns_id, name, version)
    
    new_spec = EnvSpec(
        id=full_env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs,
    )
    _check_spec_register(new_spec)
    
    if new_spec.id in registry:
        import warnings
        warnings.warn(f"Overwriting existing environment {new_spec.id}")
    registry[new_spec.id] = new_spec


def _find_spec(env_id: str) -> EnvSpec:
    """Find environment specification"""
    # Support "module:env_id" format, automatically import module
    module, env_name = (None, env_id) if ":" not in env_id else env_id.split(":", 1)
    if module is not None:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}. Failed to register environment by importing module. "
                f"Please check if '{module}' contains environment registration and can be imported."
            ) from e
    
    env_spec = registry.get(env_name)
    
    ns, name, version = parse_env_id(env_name)
    
    latest_version = find_highest_version(ns, name)
    if version is None and latest_version is not None:
        version = latest_version
        new_env_id = get_env_id(ns, name, version)
        env_spec = registry.get(new_env_id)
        if env_spec:
            import warnings
            warnings.warn(
                f"Using latest version environment `{new_env_id}` instead of unversioned environment `{env_name}`."
            )
    
    if env_spec is None:
        _check_version_exists(ns, name, version)
        raise ValueError(
            f"Registered environment not found: {env_name}. "
            f"Have you registered it, or imported a package that registers it? "
            f"Use `agenticpay.envs.pprint_registry()` to view all registered environments."
        )
    
    return env_spec


def make(
    id: str | EnvSpec,
    max_episode_steps: int | None = None,
    **kwargs: Any,
) -> BaseEnv:
    """Create environment instance
    
    Create environment instance based on environment ID or specification.
    
    Args:
        id: Environment ID string or EnvSpec object. If using string, can include module prefix, e.g., "module:Env-v0"
        max_episode_steps: Maximum steps, can override registration settings
        **kwargs: Additional parameters passed to environment constructor
        
    Returns:
        Environment instance
        
    Example:
        >>> env = make("my_env/Negotiation-v0")
        >>> env = make("my_env/Negotiation-v0", max_rounds=20)
    """
    if isinstance(id, EnvSpec):
        env_spec = id
    else:
        env_spec = _find_spec(id)
    
    # Merge parameter dictionaries
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    env_spec_kwargs.update(kwargs)
    
    # Load environment creation function
    if env_spec.entry_point is None:
        raise ValueError(f"{env_spec.id} is registered but entry_point is not specified")
    elif callable(env_spec.entry_point):
        env_creator = env_spec.entry_point
    else:
        # Assume it's a string path
        env_creator = load_env_creator(env_spec.entry_point)
    
    # Create environment instance
    try:
        env = env_creator(**env_spec_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Error creating environment {env_spec.id}, parameters: {env_spec_kwargs}"
        ) from e
    
    if not isinstance(env, BaseEnv):
        raise TypeError(
            f"Environment must inherit from BaseEnv, actual type: {type(env)}"
        )
    
    # Set environment specification
    env.unwrapped.spec = env_spec
    
    return env


def spec(env_id: str) -> EnvSpec:
    """Get environment specification
    
    Args:
        env_id: Environment ID
        
    Returns:
        Environment specification object
        
    Raises:
        ValueError: If environment does not exist
    """
    env_spec = registry.get(env_id)
    if env_spec is None:
        ns, name, version = parse_env_id(env_id)
        _check_version_exists(ns, name, version)
        raise ValueError(f"Registered environment not found: {env_id}")
    else:
        assert isinstance(
            env_spec, EnvSpec
        ), f"Type of {env_id} in registry should be `EnvSpec`, actual type is {type(env_spec)}"
        return env_spec


def pprint_registry(
    print_registry: dict[str, EnvSpec] = registry,
    *,
    num_cols: int = 3,
    exclude_namespaces: list[str] | None = None,
    disable_print: bool = False,
) -> str | None:
    """Print all registered environments
    
    Args:
        print_registry: Registry to print, defaults to global registry
        num_cols: Number of columns to display
        exclude_namespaces: List of namespaces to exclude
        disable_print: If True, return string instead of printing
        
    Returns:
        Returns string if disable_print=True, otherwise returns None
    """
    namespace_envs: dict[str, list[str]] = defaultdict(list)
    max_justify = float("-inf")
    
    for env_spec in print_registry.values():
        ns = env_spec.namespace
        
        if ns is None and isinstance(env_spec.entry_point, str):
            # Extract namespace from entry point
            env_entry_point = re.sub(r":\w+", "", env_spec.entry_point)
            split_entry_point = env_entry_point.split(".")
            
            if len(split_entry_point) >= 3:
                ns = split_entry_point[2]
            elif len(split_entry_point) > 1:
                ns = split_entry_point[1]
            else:
                ns = env_spec.name
        
        namespace_envs[ns].append(env_spec.id)
        max_justify = max(max_justify, len(env_spec.name))
    
    output: list[str] = []
    for ns, env_ids in namespace_envs.items():
        if exclude_namespaces is not None and ns in exclude_namespaces:
            continue
        
        namespace_output = f"{'=' * 5} {ns} {'=' * 5}\n"
        
        for count, env_id in enumerate(sorted(env_ids), 1):
            namespace_output += env_id.ljust(int(max_justify)) + " "
            
            if count % num_cols == 0:
                namespace_output = namespace_output.rstrip(" ")
                if count != len(env_ids):
                    namespace_output += "\n"
        
        output.append(namespace_output.rstrip(" "))
    
    if disable_print:
        return "\n".join(output)
    else:
        print("\n".join(output))

