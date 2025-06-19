#!/usr/bin/env python3
"""
Competitor Adapter Registry
===========================

Registry system for managing different serialization library adapters.
Provides a unified interface for benchmarking against various competitors.
"""

import importlib
import logging
from typing import Dict, Any, Optional, Callable, List, Set

logger = logging.getLogger(__name__)


class CompetitorAdapter:
    """Base class for competitor adapters."""
    
    def __init__(self, name: str, library_module: str, version: Optional[str] = None):
        self.name = name
        self.library_module = library_module
        self.version = version
        self.available = False
        self.module = None
        
        # Try to import the library
        try:
            self.module = importlib.import_module(library_module)
            self.available = True
            if hasattr(self.module, '__version__'):
                self.version = self.module.__version__
        except ImportError:
            logger.debug(f"Library {library_module} not available for {name}")
    
    def serialize(self, data: Any) -> Any:
        """Serialize data using this competitor."""
        raise NotImplementedError("Subclasses must implement serialize")
    
    def deserialize(self, data: Any) -> Any:
        """Deserialize data using this competitor."""
        raise NotImplementedError("Subclasses must implement deserialize")
    
    def supports_binary(self) -> bool:
        """Whether this competitor produces binary output."""
        return False
    
    def get_capabilities(self) -> Set[str]:
        """Get the data type capabilities of this adapter."""
        return {"json_safe"}  # Default: only basic JSON types


class DataSONAdapter(CompetitorAdapter):
    """DataSON adapter."""
    
    def __init__(self):
        super().__init__("datason", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced", "ml_complex"}


class DataSONAPIAdapter(CompetitorAdapter):
    """DataSON API-optimized adapter using dump_api()."""
    
    def __init__(self):
        super().__init__("datason_api", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'dump_api'):
            return self.module.dump_api(data)
        else:
            # Fallback to regular serialize for older versions
            return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'load_smart'):
            return self.module.load_smart(data)
        else:
            return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced"}


class DataSONMLAdapter(CompetitorAdapter):
    """DataSON ML-optimized adapter using dump_ml()."""
    
    def __init__(self):
        super().__init__("datason_ml", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'dump_ml'):
            return self.module.dump_ml(data)
        else:
            return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'load_smart'):
            return self.module.load_smart(data)
        else:
            return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced", "ml_complex"}


class DataSONFastAdapter(CompetitorAdapter):
    """DataSON performance-optimized adapter using dump_fast()."""
    
    def __init__(self):
        super().__init__("datason_fast", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'dump_fast'):
            return self.module.dump_fast(data)
        else:
            return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'load_smart'):
            return self.module.load_smart(data)
        else:
            return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced"}


class DataSONSecureAdapter(CompetitorAdapter):
    """DataSON security-focused adapter using dump_secure() with PII redaction."""
    
    def __init__(self):
        super().__init__("datason_secure", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'dump_secure'):
            return self.module.dump_secure(data)
        else:
            # Fallback to regular serialize for older versions
            return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'load_smart'):
            return self.module.load_smart(data)
        else:
            return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced", "ml_complex", "security_enhanced"}


class DataSONSmartAdapter(CompetitorAdapter):
    """DataSON intelligent loading adapter using load_smart() for ~90% accuracy."""
    
    def __init__(self):
        super().__init__("datason_smart", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'load_smart'):
            return self.module.load_smart(data)
        else:
            # Fallback to regular deserialize
            return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced", "ml_complex", "smart_loading"}


class DataSONPerfectAdapter(CompetitorAdapter):
    """DataSON perfect reconstruction adapter using load_perfect() for 100% accuracy."""
    
    def __init__(self):
        super().__init__("datason_perfect", "datason")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        return self.module.serialize(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("DataSON not available")
        if hasattr(self.module, 'load_perfect'):
            return self.module.load_perfect(data)
        else:
            # Fallback to regular deserialize
            return self.module.deserialize(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced", "ml_complex", "perfect_loading"}


class OrjsonAdapter(CompetitorAdapter):
    """orjson adapter - Rust-based JSON library."""
    
    def __init__(self):
        super().__init__("orjson", "orjson")
    
    def serialize(self, data: Any) -> bytes:
        if not self.available or self.module is None:
            raise RuntimeError("orjson not available")
        return self.module.dumps(data)
    
    def deserialize(self, data: bytes) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("orjson not available")
        if isinstance(data, str):
            data = data.encode()
        return self.module.loads(data)
    
    def supports_binary(self) -> bool:
        return True
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe"}


class UjsonAdapter(CompetitorAdapter):
    """ujson adapter - C-based JSON library."""
    
    def __init__(self):
        super().__init__("ujson", "ujson")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("ujson not available")
        return self.module.dumps(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("ujson not available")
        return self.module.loads(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe"}


class JsonAdapter(CompetitorAdapter):
    """Standard library json adapter."""
    
    def __init__(self):
        super().__init__("json", "json")
    
    def serialize(self, data: Any) -> str:
        if not self.available or self.module is None:
            raise RuntimeError("json not available")
        return self.module.dumps(data, default=str, ensure_ascii=False)
    
    def deserialize(self, data: str) -> Any:
        if not self.available or self.module is None:
            raise RuntimeError("json not available")
        return self.module.loads(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe"}


class PickleAdapter(CompetitorAdapter):
    """Standard library pickle adapter."""
    
    def __init__(self):
        super().__init__("pickle", "pickle")
    
    def serialize(self, data: Any) -> bytes:
        if not self.available:
            raise RuntimeError("pickle not available")
        return self.module.dumps(data)
    
    def deserialize(self, data: bytes) -> Any:
        if not self.available:
            raise RuntimeError("pickle not available")
        return self.module.loads(data)
    
    def supports_binary(self) -> bool:
        return True
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced", "ml_complex"}


class JsonpickleAdapter(CompetitorAdapter):
    """jsonpickle adapter - JSON-based object serialization."""
    
    def __init__(self):
        super().__init__("jsonpickle", "jsonpickle")
    
    def serialize(self, data: Any) -> str:
        if not self.available:
            raise RuntimeError("jsonpickle not available")
        return self.module.encode(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available:
            raise RuntimeError("jsonpickle not available")
        return self.module.decode(data)
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "object_enhanced"}


class MsgpackAdapter(CompetitorAdapter):
    """msgpack adapter - Binary serialization format."""
    
    def __init__(self):
        super().__init__("msgpack", "msgpack")
    
    def serialize(self, data: Any) -> bytes:
        if not self.available:
            raise RuntimeError("msgpack not available")
        return self.module.packb(data, default=str, strict_types=False)
    
    def deserialize(self, data: bytes) -> Any:
        if not self.available:
            raise RuntimeError("msgpack not available")
        return self.module.unpackb(data, raw=False, strict_map_key=False)
    
    def supports_binary(self) -> bool:
        return True
    
    def get_capabilities(self) -> Set[str]:
        return {"json_safe", "basic_objects"}


class CompetitorRegistry:
    """Registry for managing all competitor adapters."""
    
    def __init__(self):
        self.adapters: Dict[str, CompetitorAdapter] = {}
        self._register_adapters()
    
    def _register_adapters(self):
        """Register all available adapters."""
        adapters = [
            # DataSON variants - Phase 1 & 2 API methods
            DataSONAdapter(),
            DataSONAPIAdapter(),
            DataSONMLAdapter(),
            DataSONFastAdapter(),
            DataSONSecureAdapter(),
            DataSONSmartAdapter(),
            DataSONPerfectAdapter(),
            
            # Competitive libraries
            OrjsonAdapter(),
            UjsonAdapter(),
            JsonAdapter(),
            PickleAdapter(),
            JsonpickleAdapter(),
            MsgpackAdapter(),
        ]
        
        for adapter in adapters:
            self.adapters[adapter.name] = adapter
            logger.debug(f"Registered adapter: {adapter.name} (available: {adapter.available})")
    
    def get_available_competitors(self) -> Dict[str, Dict[str, Any]]:
        """Get all available competitors with their metadata."""
        return {
            name: {
                "version": adapter.version,
                "available": adapter.available,
                "supports_binary": adapter.supports_binary(),
                "capabilities": list(adapter.get_capabilities()),
                "module": adapter.library_module
            }
            for name, adapter in self.adapters.items()
            if adapter.available
        }
    
    def get_competitors_by_capability(self, capability: str) -> List[str]:
        """Get competitors that support a specific capability."""
        return [
            name for name, adapter in self.adapters.items()
            if adapter.available and capability in adapter.get_capabilities()
        ]
    
    def get_adapter(self, name: str) -> Optional[CompetitorAdapter]:
        """Get a specific adapter by name."""
        adapter = self.adapters.get(name)
        if adapter and adapter.available:
            return adapter
        return None
    
    def list_available_names(self) -> List[str]:
        """Get list of available competitor names."""
        return [name for name, adapter in self.adapters.items() if adapter.available]
    
    def benchmark_safe_serialize(self, adapter_name: str, data: Any) -> Optional[Any]:
        """Safely serialize data with error handling."""
        adapter = self.get_adapter(adapter_name)
        if not adapter:
            return None
        
        try:
            return adapter.serialize(data)
        except Exception as e:
            logger.warning(f"Serialization failed for {adapter_name}: {e}")
            return None
    
    def benchmark_safe_deserialize(self, adapter_name: str, serialized_data: Any) -> Optional[Any]:
        """Safely deserialize data with error handling."""
        adapter = self.get_adapter(adapter_name)
        if not adapter:
            return None
        
        try:
            return adapter.deserialize(serialized_data)
        except Exception as e:
            logger.warning(f"Deserialization failed for {adapter_name}: {e}")
            return None 