#!/usr/bin/env python3
"""
Competitor Adapter Registry
===========================

Registry system for managing different serialization library adapters.
Provides a unified interface for benchmarking against various competitors.
"""

import importlib
import logging
from typing import Dict, Any, Optional, Callable, List

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


class OrjsonAdapter(CompetitorAdapter):
    """orjson adapter - Rust-based JSON library."""
    
    def __init__(self):
        super().__init__("orjson", "orjson")
    
    def serialize(self, data: Any) -> bytes:
        if not self.available or self.module is None:
            raise RuntimeError("orjson not available")
        return self.module.dumps(data)
    
    def deserialize(self, data: bytes) -> Any:
        if not self.available:
            raise RuntimeError("orjson not available")
        if isinstance(data, str):
            data = data.encode()
        return self.module.loads(data)
    
    def supports_binary(self) -> bool:
        return True


class UjsonAdapter(CompetitorAdapter):
    """ujson adapter - C-based JSON library."""
    
    def __init__(self):
        super().__init__("ujson", "ujson")
    
    def serialize(self, data: Any) -> str:
        if not self.available:
            raise RuntimeError("ujson not available")
        return self.module.dumps(data)
    
    def deserialize(self, data: str) -> Any:
        if not self.available:
            raise RuntimeError("ujson not available")
        return self.module.loads(data)


class JsonAdapter(CompetitorAdapter):
    """Standard library json adapter."""
    
    def __init__(self):
        super().__init__("json", "json")
    
    def serialize(self, data: Any) -> str:
        if not self.available:
            raise RuntimeError("json not available")
        return self.module.dumps(data, default=str, ensure_ascii=False)
    
    def deserialize(self, data: str) -> Any:
        if not self.available:
            raise RuntimeError("json not available")
        return self.module.loads(data)


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


class CompetitorRegistry:
    """Registry for managing all competitor adapters."""
    
    def __init__(self):
        self.adapters: Dict[str, CompetitorAdapter] = {}
        self._register_adapters()
    
    def _register_adapters(self):
        """Register all available adapters."""
        adapter_classes = [
            DataSONAdapter,
            OrjsonAdapter, 
            UjsonAdapter,
            JsonAdapter,
            PickleAdapter,
            JsonpickleAdapter,
            MsgpackAdapter
        ]
        
        for adapter_class in adapter_classes:
            try:
                adapter = adapter_class()
                self.adapters[adapter.name] = adapter
                if adapter.available:
                    logger.info(f"✅ {adapter.name} available (v{adapter.version})")
                else:
                    logger.debug(f"❌ {adapter.name} not available")
            except Exception as e:
                logger.warning(f"Failed to initialize {adapter_class.__name__}: {e}")
    
    def get_available_competitors(self) -> Dict[str, Dict[str, Any]]:
        """Get all available competitors with their metadata."""
        return {
            name: {
                "version": adapter.version,
                "available": adapter.available,
                "supports_binary": adapter.supports_binary(),
                "module": adapter.library_module
            }
            for name, adapter in self.adapters.items()
            if adapter.available
        }
    
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