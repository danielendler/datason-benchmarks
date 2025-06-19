#!/usr/bin/env python3
"""
Phase 4 Decision Engine
=======================

Intelligent decision support system for optimal serialization library selection
based on use case requirements, performance data, and domain-specific needs.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class Priority(Enum):
    """User priority levels for different criteria."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Domain(Enum):
    """Domain categories for specialized recommendations."""
    WEB_API = "web_api"
    MACHINE_LEARNING = "machine_learning"
    DATA_ENGINEERING = "data_engineering"
    FINANCIAL_SERVICES = "financial_services"
    ENTERPRISE = "enterprise"
    HIGH_PERFORMANCE = "high_performance"
    RESEARCH = "research"
    GENERAL = "general"


@dataclass
class UserRequirements:
    """User requirements for serialization library selection."""
    domain: Domain
    speed_priority: Priority
    accuracy_priority: Priority
    security_priority: Priority
    compatibility_priority: Priority
    data_types: List[str]  # e.g., ['datetime', 'uuid', 'decimal', 'numpy']
    volume_level: str  # 'low', 'medium', 'high'
    team_expertise: str  # 'beginner', 'intermediate', 'expert'
    existing_stack: List[str]  # e.g., ['django', 'pandas', 'numpy']
    compliance_needs: bool
    performance_budget_ms: Optional[float] = None


@dataclass
class LibraryScore:
    """Scoring result for a library recommendation."""
    library_name: str
    method_name: Optional[str]
    total_score: float
    criteria_scores: Dict[str, float]
    pros: List[str]
    cons: List[str]
    confidence: float
    use_case_fit: str  # 'excellent', 'good', 'fair', 'poor'


class DecisionEngine:
    """Phase 4 intelligent decision engine for library selection."""
    
    def __init__(self, benchmark_data_dir: str = "data/results"):
        self.benchmark_data_dir = Path(benchmark_data_dir)
        
        # Load performance baselines from benchmark data
        self.performance_baselines = self._load_performance_baselines()
        
        # Define library capabilities matrix
        self.capability_matrix = self._define_capability_matrix()
        
        # Define scoring weights
        self.scoring_weights = self._define_scoring_weights()
    
    def recommend_library(self, requirements: UserRequirements) -> List[LibraryScore]:
        """Get ranked library recommendations based on user requirements."""
        logger.info(f"🎯 Generating recommendations for {requirements.domain.value} domain")
        
        # Get all available libraries/methods
        candidates = self._get_available_candidates(requirements)
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, requirements)
            if score:
                scored_candidates.append(score)
        
        # Sort by total score (descending)
        scored_candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        # Add confidence and use case fit assessment
        for candidate in scored_candidates:
            candidate.confidence = self._calculate_confidence(candidate, requirements)
            candidate.use_case_fit = self._assess_use_case_fit(candidate, requirements)
        
        logger.info(f"✅ Generated {len(scored_candidates)} recommendations")
        return scored_candidates
    
    def explain_recommendation(self, recommendation: LibraryScore, 
                             requirements: UserRequirements) -> Dict[str, Any]:
        """Provide detailed explanation for a recommendation."""
        explanation = {
            "library": recommendation.library_name,
            "method": recommendation.method_name,
            "score": recommendation.total_score,
            "confidence": recommendation.confidence,
            "fit_assessment": recommendation.use_case_fit,
            "reasoning": {
                "strengths": recommendation.pros,
                "limitations": recommendation.cons,
                "criteria_analysis": {}
            },
            "implementation_guidance": self._get_implementation_guidance(
                recommendation, requirements
            ),
            "performance_expectations": self._get_performance_expectations(
                recommendation, requirements
            ),
            "alternatives": self._get_alternatives(recommendation, requirements)
        }
        
        # Detailed criteria analysis
        for criteria, score in recommendation.criteria_scores.items():
            explanation["reasoning"]["criteria_analysis"][criteria] = {
                "score": score,
                "interpretation": self._interpret_criteria_score(criteria, score),
                "importance": self._get_criteria_importance(criteria, requirements)
            }
        
        return explanation
    
    def compare_libraries(self, libraries: List[str], 
                         requirements: UserRequirements) -> Dict[str, Any]:
        """Compare specific libraries against requirements."""
        comparison = {
            "comparison_criteria": [],
            "libraries": {},
            "summary": {},
            "decision_matrix": self._create_decision_matrix(libraries, requirements)
        }
        
        # Score each library
        for library in libraries:
            candidate = {"library": library, "method": None}
            score = self._score_candidate(candidate, requirements)
            
            if score:
                comparison["libraries"][library] = {
                    "score": score.total_score,
                    "criteria_scores": score.criteria_scores,
                    "pros": score.pros,
                    "cons": score.cons,
                    "recommended_for": self._get_recommended_scenarios(library, requirements)
                }
        
        # Generate comparison summary
        if comparison["libraries"]:
            best_library = max(comparison["libraries"].items(), 
                             key=lambda x: x[1]["score"])
            comparison["summary"]["recommended"] = best_library[0]
            comparison["summary"]["reasoning"] = f"Highest overall score for {requirements.domain.value} use case"
        
        return comparison
    
    def _load_performance_baselines(self) -> Dict[str, Any]:
        """Load performance baselines from benchmark results."""
        baselines = {
            "serialization": {},
            "deserialization": {},
            "success_rates": {},
            "accuracy_scores": {}
        }
        
        # Try to load latest comprehensive benchmark results
        result_files = list(self.benchmark_data_dir.glob("*complete*.json"))
        if not result_files:
            result_files = list(self.benchmark_data_dir.glob("*competitive*.json"))
        
        if result_files:
            # Use most recent file
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            
            try:
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                # Extract performance data
                if "competitive" in results and "tiers" in results["competitive"]:
                    tiers = results["competitive"]["tiers"]
                    
                    for tier_name, tier_data in tiers.items():
                        if "datasets" in tier_data:
                            for dataset_name, dataset in tier_data["datasets"].items():
                                # Serialization performance
                                if "serialization" in dataset:
                                    for library, perf in dataset["serialization"].items():
                                        if isinstance(perf, dict) and "mean_ms" in perf:
                                            key = f"{tier_name}_{dataset_name}"
                                            if key not in baselines["serialization"]:
                                                baselines["serialization"][key] = {}
                                            baselines["serialization"][key][library] = perf["mean_ms"]
                
                logger.info(f"📊 Loaded performance baselines from {latest_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load performance baselines: {e}")
        
        return baselines
    
    def _define_capability_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Define capability matrix for all libraries."""
        return {
            "datason": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": True,
                "pandas": True,
                "complex_objects": True,
                "custom_types": True,
                "security_features": True,
                "json_compatibility": True,
                "binary_format": False
            },
            "datason_api": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": False,
                "pandas": False,
                "complex_objects": True,
                "custom_types": True,
                "security_features": False,
                "json_compatibility": True,
                "binary_format": False
            },
            "datason_ml": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": True,
                "pandas": True,
                "complex_objects": True,
                "custom_types": True,
                "security_features": False,
                "json_compatibility": False,
                "binary_format": False
            },
            "datason_fast": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": True,
                "pandas": True,
                "complex_objects": True,
                "custom_types": False,
                "security_features": False,
                "json_compatibility": False,
                "binary_format": False
            },
            "datason_secure": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": False,
                "pandas": False,
                "complex_objects": True,
                "custom_types": True,
                "security_features": True,
                "json_compatibility": True,
                "binary_format": False
            },
            "orjson": {
                "datetime": False,
                "uuid": False,
                "decimal": False,
                "numpy": False,
                "pandas": False,
                "complex_objects": False,
                "custom_types": False,
                "security_features": False,
                "json_compatibility": True,
                "binary_format": False
            },
            "ujson": {
                "datetime": False,
                "uuid": False,
                "decimal": False,
                "numpy": False,
                "pandas": False,
                "complex_objects": False,
                "custom_types": False,
                "security_features": False,
                "json_compatibility": True,
                "binary_format": False
            },
            "pickle": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": True,
                "pandas": True,
                "complex_objects": True,
                "custom_types": True,
                "security_features": False,
                "json_compatibility": False,
                "binary_format": True
            },
            "jsonpickle": {
                "datetime": True,
                "uuid": True,
                "decimal": True,
                "numpy": True,
                "pandas": True,
                "complex_objects": True,
                "custom_types": True,
                "security_features": False,
                "json_compatibility": True,
                "binary_format": False
            }
        }
    
    def _define_scoring_weights(self) -> Dict[str, Dict[Priority, float]]:
        """Define scoring weights based on priority levels."""
        return {
            "speed": {
                Priority.LOW: 0.1,
                Priority.MEDIUM: 0.25,
                Priority.HIGH: 0.4,
                Priority.CRITICAL: 0.6
            },
            "accuracy": {
                Priority.LOW: 0.1,
                Priority.MEDIUM: 0.2,
                Priority.HIGH: 0.35,
                Priority.CRITICAL: 0.5
            },
            "security": {
                Priority.LOW: 0.05,
                Priority.MEDIUM: 0.15,
                Priority.HIGH: 0.3,
                Priority.CRITICAL: 0.45
            },
            "compatibility": {
                Priority.LOW: 0.1,
                Priority.MEDIUM: 0.2,
                Priority.HIGH: 0.3,
                Priority.CRITICAL: 0.4
            }
        }
    
    def _get_available_candidates(self, requirements: UserRequirements) -> List[Dict[str, Any]]:
        """Get available library/method candidates based on requirements."""
        candidates = []
        
        # DataSON variants
        if requirements.domain == Domain.WEB_API:
            candidates.extend([
                {"library": "datason", "method": "dump_api"},
                {"library": "datason", "method": "serialize"}
            ])
        elif requirements.domain == Domain.MACHINE_LEARNING:
            candidates.extend([
                {"library": "datason", "method": "dump_ml"},
                {"library": "datason", "method": "serialize"}
            ])
        elif requirements.domain == Domain.FINANCIAL_SERVICES:
            candidates.extend([
                {"library": "datason", "method": "dump_secure"},
                {"library": "datason", "method": "serialize"}
            ])
        elif requirements.domain == Domain.HIGH_PERFORMANCE:
            candidates.extend([
                {"library": "datason", "method": "dump_fast"},
                {"library": "datason", "method": "serialize"}
            ])
        else:
            candidates.extend([
                {"library": "datason", "method": "serialize"},
                {"library": "datason", "method": "dump_api"},
                {"library": "datason", "method": "dump_fast"}
            ])
        
        # Standard JSON libraries (for basic JSON compatibility)
        if "datetime" not in requirements.data_types and "uuid" not in requirements.data_types:
            candidates.extend([
                {"library": "orjson", "method": None},
                {"library": "ujson", "method": None}
            ])
        
        # Pickle for binary/Python-specific needs
        if requirements.compatibility_priority != Priority.CRITICAL:  # JSON compatibility not critical
            candidates.append({"library": "pickle", "method": None})
        
        # jsonpickle for complex objects with JSON compatibility
        if any(dt in requirements.data_types for dt in ["numpy", "pandas", "custom_objects"]):
            candidates.append({"library": "jsonpickle", "method": None})
        
        return candidates
    
    def _score_candidate(self, candidate: Dict[str, Any], 
                        requirements: UserRequirements) -> Optional[LibraryScore]:
        """Score a single candidate library/method."""
        library = candidate["library"]
        method = candidate.get("method")
        
        # Check if library supports required data types
        if not self._supports_required_types(library, requirements.data_types):
            return None
        
        scores = {}
        pros = []
        cons = []
        
        # Speed scoring
        scores["speed"] = self._score_speed(library, method, requirements)
        if scores["speed"] > 0.8:
            pros.append(f"Excellent speed performance")
        elif scores["speed"] < 0.3:
            cons.append(f"Below-average speed")
        
        # Accuracy scoring
        scores["accuracy"] = self._score_accuracy(library, method, requirements)
        if scores["accuracy"] > 0.9:
            pros.append(f"Excellent data preservation")
        elif scores["accuracy"] < 0.5:
            cons.append(f"Limited data type support")
        
        # Security scoring
        scores["security"] = self._score_security(library, method, requirements)
        if scores["security"] > 0.8 and requirements.security_priority.value >= 3:
            pros.append(f"Strong security features")
        elif scores["security"] < 0.3 and requirements.security_priority.value >= 3:
            cons.append(f"Limited security features")
        
        # Compatibility scoring
        scores["compatibility"] = self._score_compatibility(library, method, requirements)
        if scores["compatibility"] > 0.9:
            pros.append(f"Excellent compatibility")
        elif scores["compatibility"] < 0.4:
            cons.append(f"Limited compatibility")
        
        # Calculate weighted total score
        weights = self.scoring_weights
        total_score = (
            scores["speed"] * weights["speed"][requirements.speed_priority] +
            scores["accuracy"] * weights["accuracy"][requirements.accuracy_priority] +
            scores["security"] * weights["security"][requirements.security_priority] +
            scores["compatibility"] * weights["compatibility"][requirements.compatibility_priority]
        )
        
        # Normalize by total possible weight
        max_weight = sum([
            weights["speed"][requirements.speed_priority],
            weights["accuracy"][requirements.accuracy_priority],
            weights["security"][requirements.security_priority],
            weights["compatibility"][requirements.compatibility_priority]
        ])
        
        total_score = total_score / max_weight
        
        # Add domain-specific bonuses/penalties
        total_score = self._apply_domain_adjustments(total_score, library, method, requirements)
        
        return LibraryScore(
            library_name=library,
            method_name=method,
            total_score=total_score,
            criteria_scores=scores,
            pros=pros,
            cons=cons,
            confidence=0.0,  # Will be calculated later
            use_case_fit=""   # Will be calculated later
        )
    
    def _supports_required_types(self, library: str, data_types: List[str]) -> bool:
        """Check if library supports all required data types."""
        if library not in self.capability_matrix:
            return False
        
        capabilities = self.capability_matrix[library]
        
        for data_type in data_types:
            if data_type in capabilities and not capabilities[data_type]:
                return False
        
        return True
    
    def _score_speed(self, library: str, method: Optional[str], 
                    requirements: UserRequirements) -> float:
        """Score library speed performance."""
        # Base speed scores (based on typical performance characteristics)
        base_scores = {
            "orjson": 0.95,
            "ujson": 0.90,
            "datason_fast": 0.85,
            "datason": 0.75,
            "datason_api": 0.70,
            "datason_ml": 0.70,
            "datason_secure": 0.55,  # Security overhead
            "pickle": 0.60,
            "jsonpickle": 0.40
        }
        
        # Use method-specific library name if available
        library_key = f"{library}_{method}" if method else library
        if library_key not in base_scores:
            library_key = library
        
        base_score = base_scores.get(library_key, 0.5)
        
        # Adjust based on volume level
        if requirements.volume_level == "high" and library in ["orjson", "ujson", "datason_fast"]:
            base_score += 0.1
        elif requirements.volume_level == "high" and library in ["jsonpickle"]:
            base_score -= 0.2
        
        return min(1.0, max(0.0, base_score))
    
    def _score_accuracy(self, library: str, method: Optional[str], 
                       requirements: UserRequirements) -> float:
        """Score library accuracy and data preservation."""
        capabilities = self.capability_matrix.get(library, {})
        
        # Count supported required data types
        supported_types = 0
        total_required = len(requirements.data_types)
        
        if total_required == 0:
            return 1.0  # No specific requirements
        
        for data_type in requirements.data_types:
            if capabilities.get(data_type, False):
                supported_types += 1
        
        base_score = supported_types / total_required
        
        # Bonus for libraries known for high accuracy
        if library.startswith("datason") or library == "pickle":
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _score_security(self, library: str, method: Optional[str], 
                       requirements: UserRequirements) -> float:
        """Score library security features."""
        capabilities = self.capability_matrix.get(library, {})
        
        # Base security score
        if capabilities.get("security_features", False):
            base_score = 0.9
        elif library == "pickle":
            base_score = 0.1  # Pickle has security risks
        else:
            base_score = 0.5  # Neutral
        
        # Method-specific adjustments
        if method == "dump_secure":
            base_score = 1.0
        
        # Compliance needs adjustment
        if requirements.compliance_needs and not capabilities.get("security_features", False):
            base_score -= 0.3
        
        return min(1.0, max(0.0, base_score))
    
    def _score_compatibility(self, library: str, method: Optional[str], 
                           requirements: UserRequirements) -> float:
        """Score library compatibility and integration ease."""
        capabilities = self.capability_matrix.get(library, {})
        
        # JSON compatibility scoring
        json_compat_score = 1.0 if capabilities.get("json_compatibility", False) else 0.3
        
        # Existing stack compatibility
        stack_bonus = 0.0
        if "django" in requirements.existing_stack and library.startswith("datason"):
            stack_bonus += 0.1
        if any(ml in requirements.existing_stack for ml in ["pandas", "numpy"]) and \
           (library.startswith("datason") or library == "pickle"):
            stack_bonus += 0.1
        
        # Team expertise adjustment
        if requirements.team_expertise == "beginner":
            if library in ["orjson", "ujson"] or library.startswith("datason"):
                stack_bonus += 0.1
            elif library in ["pickle", "jsonpickle"]:
                stack_bonus -= 0.1
        
        base_score = json_compat_score + stack_bonus
        return min(1.0, max(0.0, base_score))
    
    def _apply_domain_adjustments(self, base_score: float, library: str, 
                                method: Optional[str], requirements: UserRequirements) -> float:
        """Apply domain-specific score adjustments."""
        adjustment = 0.0
        
        if requirements.domain == Domain.WEB_API:
            if method == "dump_api":
                adjustment += 0.15
            elif library in ["orjson", "ujson"]:
                adjustment += 0.1
            elif library == "pickle":
                adjustment -= 0.2  # Not web-friendly
        
        elif requirements.domain == Domain.MACHINE_LEARNING:
            if method == "dump_ml":
                adjustment += 0.15
            elif library == "pickle":
                adjustment += 0.1  # Common in ML
            elif library in ["orjson", "ujson"]:
                adjustment -= 0.1  # Limited ML support
        
        elif requirements.domain == Domain.FINANCIAL_SERVICES:
            if method == "dump_secure":
                adjustment += 0.2
            elif requirements.compliance_needs and not self.capability_matrix.get(library, {}).get("security_features", False):
                adjustment -= 0.15
        
        elif requirements.domain == Domain.HIGH_PERFORMANCE:
            if method == "dump_fast":
                adjustment += 0.15
            elif library in ["orjson", "ujson"]:
                adjustment += 0.1
            elif library == "jsonpickle":
                adjustment -= 0.15
        
        return min(1.0, max(0.0, base_score + adjustment))
    
    def _calculate_confidence(self, score: LibraryScore, requirements: UserRequirements) -> float:
        """Calculate confidence level in the recommendation."""
        confidence_factors = []
        
        # Score-based confidence
        if score.total_score > 0.8:
            confidence_factors.append(0.3)
        elif score.total_score > 0.6:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Data availability confidence
        if score.library_name in self.performance_baselines.get("serialization", {}):
            confidence_factors.append(0.2)
        
        # Domain expertise confidence
        domain_expertise = {
            Domain.WEB_API: ["datason_api", "orjson", "ujson"],
            Domain.MACHINE_LEARNING: ["datason_ml", "pickle", "datason"],
            Domain.FINANCIAL_SERVICES: ["datason_secure", "datason"],
            Domain.HIGH_PERFORMANCE: ["datason_fast", "orjson", "ujson"]
        }
        
        if score.library_name in domain_expertise.get(requirements.domain, []):
            confidence_factors.append(0.3)
        
        # Requirements clarity confidence
        if len(requirements.data_types) > 0:
            confidence_factors.append(0.2)
        
        return min(1.0, sum(confidence_factors))
    
    def _assess_use_case_fit(self, score: LibraryScore, requirements: UserRequirements) -> str:
        """Assess how well the library fits the use case."""
        if score.total_score > 0.8 and score.confidence > 0.7:
            return "excellent"
        elif score.total_score > 0.6 and score.confidence > 0.5:
            return "good"
        elif score.total_score > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _get_implementation_guidance(self, recommendation: LibraryScore, 
                                   requirements: UserRequirements) -> Dict[str, Any]:
        """Get implementation guidance for the recommended library."""
        guidance = {
            "installation": f"pip install {recommendation.library_name}",
            "basic_usage": "",
            "best_practices": [],
            "common_pitfalls": []
        }
        
        if recommendation.library_name.startswith("datason"):
            guidance["basic_usage"] = """
import datason

# Serialization
data = {"timestamp": datetime.now(), "id": uuid4()}
serialized = datason.serialize(data)

# Deserialization  
restored = datason.deserialize(serialized)
"""
            if recommendation.method_name:
                guidance["basic_usage"] = f"""
import datason

# Using {recommendation.method_name}
data = {{"key": "value"}}
serialized = datason.{recommendation.method_name}(data)
"""
            
            guidance["best_practices"] = [
                "Use type hints for better serialization accuracy",
                "Handle datetime objects with timezone awareness",
                "Consider using specific dump methods for optimized performance"
            ]
            
        elif recommendation.library_name in ["orjson", "ujson"]:
            guidance["basic_usage"] = f"""
import {recommendation.library_name}

# Serialization
data = {{"key": "value"}}
serialized = {recommendation.library_name}.dumps(data)

# Deserialization
restored = {recommendation.library_name}.loads(serialized)
"""
            
            guidance["common_pitfalls"] = [
                "Does not handle datetime/UUID objects natively",
                "Convert complex objects to JSON-compatible types first"
            ]
        
        return guidance
    
    def _get_performance_expectations(self, recommendation: LibraryScore, 
                                    requirements: UserRequirements) -> Dict[str, Any]:
        """Get performance expectations for the recommendation."""
        expectations = {
            "typical_speed": "Unknown",
            "memory_usage": "Unknown", 
            "scalability": "Unknown",
            "bottlenecks": []
        }
        
        # Speed expectations based on library characteristics
        if recommendation.library_name in ["orjson", "ujson"]:
            expectations["typical_speed"] = "Very fast (0.01-0.02ms for small objects)"
            expectations["scalability"] = "Excellent for high-volume JSON"
        elif recommendation.library_name.startswith("datason"):
            if recommendation.method_name == "dump_fast":
                expectations["typical_speed"] = "Fast (0.02-0.05ms for small objects)"
            else:
                expectations["typical_speed"] = "Moderate (0.03-0.08ms for small objects)"
            expectations["scalability"] = "Good for complex object serialization"
        elif recommendation.library_name == "pickle":
            expectations["typical_speed"] = "Moderate (0.05-0.15ms for small objects)"
            expectations["scalability"] = "Good for Python-specific data"
        
        # Add performance warnings if needed
        if requirements.volume_level == "high" and recommendation.library_name == "jsonpickle":
            expectations["bottlenecks"].append("May be slow for high-volume serialization")
        
        if requirements.security_priority.value >= 3 and recommendation.method_name == "dump_secure":
            expectations["bottlenecks"].append("Security processing adds ~25% overhead")
        
        return expectations
    
    def _get_alternatives(self, recommendation: LibraryScore, 
                         requirements: UserRequirements) -> List[Dict[str, str]]:
        """Get alternative recommendations."""
        alternatives = []
        
        # If recommending DataSON, suggest other DataSON methods
        if recommendation.library_name.startswith("datason"):
            if recommendation.method_name != "dump_api":
                alternatives.append({
                    "library": "datason",
                    "method": "dump_api", 
                    "reason": "For better JSON compatibility"
                })
            if recommendation.method_name != "dump_fast":
                alternatives.append({
                    "library": "datason",
                    "method": "dump_fast",
                    "reason": "For higher performance"
                })
        
        # If recommending non-DataSON, suggest DataSON alternative
        else:
            if requirements.domain == Domain.WEB_API:
                alternatives.append({
                    "library": "datason",
                    "method": "dump_api",
                    "reason": "For better object handling with JSON compatibility"
                })
            elif requirements.domain == Domain.MACHINE_LEARNING:
                alternatives.append({
                    "library": "datason", 
                    "method": "dump_ml",
                    "reason": "For native ML framework support"
                })
        
        return alternatives
    
    def _interpret_criteria_score(self, criteria: str, score: float) -> str:
        """Interpret criteria score for explanation."""
        if score > 0.8:
            return f"Excellent {criteria} characteristics"
        elif score > 0.6:
            return f"Good {criteria} performance"
        elif score > 0.4:
            return f"Adequate {criteria} capabilities"
        else:
            return f"Limited {criteria} support"
    
    def _get_criteria_importance(self, criteria: str, requirements: UserRequirements) -> str:
        """Get importance level of criteria for requirements."""
        importance_map = {
            "speed": requirements.speed_priority.name,
            "accuracy": requirements.accuracy_priority.name,
            "security": requirements.security_priority.name,
            "compatibility": requirements.compatibility_priority.name
        }
        
        return importance_map.get(criteria, "MEDIUM")
    
    def _create_decision_matrix(self, libraries: List[str], 
                              requirements: UserRequirements) -> Dict[str, Any]:
        """Create decision matrix for library comparison."""
        matrix = {
            "criteria": ["Speed", "Accuracy", "Security", "Compatibility"],
            "weights": {
                "Speed": self.scoring_weights["speed"][requirements.speed_priority],
                "Accuracy": self.scoring_weights["accuracy"][requirements.accuracy_priority],
                "Security": self.scoring_weights["security"][requirements.security_priority],
                "Compatibility": self.scoring_weights["compatibility"][requirements.compatibility_priority]
            },
            "scores": {}
        }
        
        for library in libraries:
            candidate = {"library": library, "method": None}
            score = self._score_candidate(candidate, requirements)
            
            if score:
                matrix["scores"][library] = {
                    "Speed": score.criteria_scores["speed"],
                    "Accuracy": score.criteria_scores["accuracy"],
                    "Security": score.criteria_scores["security"],
                    "Compatibility": score.criteria_scores["compatibility"],
                    "Weighted_Total": score.total_score
                }
        
        return matrix
    
    def _get_recommended_scenarios(self, library: str, requirements: UserRequirements) -> List[str]:
        """Get scenarios where this library is recommended."""
        scenarios = []
        
        if library.startswith("datason"):
            scenarios.append("Complex object serialization")
            scenarios.append("Type-safe data persistence")
            
            if library.endswith("_api"):
                scenarios.append("Web API development")
            elif library.endswith("_ml"):
                scenarios.append("Machine learning workflows")
            elif library.endswith("_secure"):
                scenarios.append("Security-sensitive applications")
            elif library.endswith("_fast"):
                scenarios.append("High-performance applications")
        
        elif library in ["orjson", "ujson"]:
            scenarios.append("High-speed JSON processing")
            scenarios.append("Simple data structures")
            scenarios.append("Web service APIs")
        
        elif library == "pickle":
            scenarios.append("Python-specific data structures")
            scenarios.append("Internal application communication")
            scenarios.append("Machine learning model serialization")
        
        return scenarios


def create_requirements_from_questionnaire(answers: Dict[str, Any]) -> UserRequirements:
    """Create UserRequirements from questionnaire answers."""
    # Map questionnaire answers to requirements
    domain_map = {
        "web": Domain.WEB_API,
        "ml": Domain.MACHINE_LEARNING,
        "data": Domain.DATA_ENGINEERING,
        "finance": Domain.FINANCIAL_SERVICES,
        "enterprise": Domain.ENTERPRISE,
        "performance": Domain.HIGH_PERFORMANCE,
        "general": Domain.GENERAL
    }
    
    priority_map = {
        "low": Priority.LOW,
        "medium": Priority.MEDIUM,
        "high": Priority.HIGH,
        "critical": Priority.CRITICAL
    }
    
    return UserRequirements(
        domain=domain_map.get(answers.get("domain", "general"), Domain.GENERAL),
        speed_priority=priority_map.get(answers.get("speed_priority", "medium"), Priority.MEDIUM),
        accuracy_priority=priority_map.get(answers.get("accuracy_priority", "high"), Priority.HIGH),
        security_priority=priority_map.get(answers.get("security_priority", "medium"), Priority.MEDIUM),
        compatibility_priority=priority_map.get(answers.get("compatibility_priority", "high"), Priority.HIGH),
        data_types=answers.get("data_types", []),
        volume_level=answers.get("volume_level", "medium"),
        team_expertise=answers.get("team_expertise", "intermediate"),
        existing_stack=answers.get("existing_stack", []),
        compliance_needs=answers.get("compliance_needs", False),
        performance_budget_ms=answers.get("performance_budget_ms")
    )


def main():
    """CLI entry point for the decision engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DataSON Decision Engine')
    parser.add_argument('--domain', required=True,
                       choices=['web', 'ml', 'data', 'finance', 'enterprise', 'performance', 'general'],
                       help='Application domain')
    parser.add_argument('--speed-priority', choices=['low', 'medium', 'high', 'critical'],
                       default='medium', help='Speed priority level')
    parser.add_argument('--accuracy-priority', choices=['low', 'medium', 'high', 'critical'],
                       default='high', help='Accuracy priority level')
    parser.add_argument('--security-priority', choices=['low', 'medium', 'high', 'critical'],
                       default='medium', help='Security priority level')
    parser.add_argument('--compatibility-priority', choices=['low', 'medium', 'high', 'critical'],
                       default='high', help='Compatibility priority level')
    parser.add_argument('--data-types', nargs='*', default=[],
                       help='Required data types (datetime, uuid, decimal, numpy, pandas)')
    parser.add_argument('--volume', choices=['low', 'medium', 'high'], default='medium',
                       help='Data volume level')
    parser.add_argument('--compare', nargs='*', help='Compare specific libraries')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create requirements from CLI args
    questionnaire_answers = {
        "domain": args.domain,
        "speed_priority": args.speed_priority,
        "accuracy_priority": args.accuracy_priority,
        "security_priority": args.security_priority,
        "compatibility_priority": args.compatibility_priority,
        "data_types": args.data_types,
        "volume_level": args.volume,
        "team_expertise": "intermediate",
        "existing_stack": [],
        "compliance_needs": args.security_priority in ['high', 'critical']
    }
    
    requirements = create_requirements_from_questionnaire(questionnaire_answers)
    
    # Initialize decision engine
    engine = DecisionEngine()
    
    try:
        if args.compare:
            # Compare specific libraries
            comparison = engine.compare_libraries(args.compare, requirements)
            print(f"🔍 Library Comparison for {args.domain} domain:")
            
            for library, data in comparison["libraries"].items():
                print(f"\n📚 {library}:")
                print(f"  Score: {data['score']:.2f}")
                print(f"  Pros: {', '.join(data['pros'][:2])}")
                
            if "recommended" in comparison["summary"]:
                print(f"\n🏆 Recommended: {comparison['summary']['recommended']}")
        
        else:
            # Get recommendations
            recommendations = engine.recommend_library(requirements)
            
            if recommendations:
                print(f"🎯 Top Recommendations for {args.domain} domain:")
                
                for i, rec in enumerate(recommendations[:3], 1):
                    method_str = f".{rec.method_name}" if rec.method_name else ""
                    print(f"\n{i}. {rec.library_name}{method_str}")
                    print(f"   Score: {rec.total_score:.2f} | Confidence: {rec.confidence:.2f}")
                    print(f"   Fit: {rec.use_case_fit}")
                    if rec.pros:
                        print(f"   ✅ {rec.pros[0]}")
                    if rec.cons:
                        print(f"   ⚠️  {rec.cons[0]}")
                
                # Detailed explanation for top recommendation
                if len(recommendations) > 0:
                    explanation = engine.explain_recommendation(recommendations[0], requirements)
                    print(f"\n📋 Why {recommendations[0].library_name}?")
                    print(f"   {explanation['implementation_guidance']['basic_usage'][:100]}...")
            
            else:
                print("❌ No suitable recommendations found for your requirements")
        
        return 0
        
    except Exception as e:
        logger.error(f"Decision engine failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 