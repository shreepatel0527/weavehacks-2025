"""
Chemistry calculation utilities for nanoparticle synthesis with error handling
"""
import weave
import wandb
from typing import Dict, Optional, Union
from config.settings import get_chemistry_config
from utils.error_handling import (
    CalculationError, safe_execute, validate_input,
    ErrorCategory, ErrorSeverity
)

def safe_wandb_log(data: dict):
    """Safely log to wandb, handling cases where wandb is not initialized"""
    try:
        safe_wandb_log(data)
    except wandb.errors.UsageError:
        # wandb not initialized, try to initialize minimally
        try:
            wandb.init(project="lab-assistant-calculations", mode="disabled")
            safe_wandb_log(data)
        except Exception:
            # If all else fails, just skip logging
            pass
    except Exception:
        # Any other wandb error, skip logging
        pass

# Get configuration
config = get_chemistry_config()

@weave.op()
@safe_execute
@validate_input({
    'mass_g': lambda x: x >= 0,
    'molecular_weight': lambda x: x > 0
})
def calculate_moles(mass_g: float, molecular_weight: float) -> float:
    """
    Calculate moles from mass and molecular weight
    
    Args:
        mass_g: Mass in grams
        molecular_weight: Molecular weight in g/mol
    
    Returns:
        Number of moles
    
    Raises:
        CalculationError: If inputs are invalid
    """
    if molecular_weight == 0:
        raise CalculationError(
            "Molecular weight cannot be zero",
            "calculate_moles",
            {'mass_g': mass_g, 'molecular_weight': molecular_weight}
        )
    
    return mass_g / molecular_weight

@weave.op()
@safe_execute
@validate_input({
    'moles': lambda x: x >= 0,
    'molecular_weight': lambda x: x > 0
})
def calculate_mass(moles: float, molecular_weight: float) -> float:
    """
    Calculate mass from moles and molecular weight
    
    Args:
        moles: Number of moles
        molecular_weight: Molecular weight in g/mol
    
    Returns:
        Mass in grams
    """
    return moles * molecular_weight

@weave.op()
@safe_execute
@validate_input({
    'gold_mass_g': lambda x: x > 0,
    'equivalents': lambda x: x > 0
})
def calculate_sulfur_amount(gold_mass_g: float, 
                          equivalents: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate the amount of 2-phenylethanethiol needed
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
        equivalents: Number of molar equivalents (uses config default if None)
    
    Returns:
        dict: Contains calculated values with units
    
    Raises:
        CalculationError: If calculation fails
    """
    if equivalents is None:
        equivalents = config.DEFAULT_EQUIVALENTS_SULFUR
    
    try:
        moles_gold = calculate_moles(gold_mass_g, config.MW_HAuCl4_3H2O)
        moles_sulfur = moles_gold * equivalents
        mass_sulfur = calculate_mass(moles_sulfur, config.MW_PhCH2CH2SH)
        
        # Calculate volume needed (assuming density ~1.1 g/mL for PhCH2CH2SH)
        volume_sulfur_ml = mass_sulfur / 1.1
        
        result = {
            'moles_gold': round(moles_gold, 6),
            'moles_sulfur': round(moles_sulfur, 6),
            'mass_sulfur_g': round(mass_sulfur, 4),
            'volume_sulfur_ml': round(volume_sulfur_ml, 2),
            'equivalents': equivalents,
            'units': {
                'moles_gold': 'mol',
                'moles_sulfur': 'mol',
                'mass_sulfur_g': 'g',
                'volume_sulfur_ml': 'mL'
            }
        }
        
        # Log calculation
        safe_wandb_log({
            'calculation': {
                'type': 'sulfur_amount',
                'input': gold_mass_g,
                'output': mass_sulfur,
                'equivalents': equivalents
            }
        })
        
        return result
        
    except Exception as e:
        raise CalculationError(
            f"Failed to calculate sulfur amount: {str(e)}",
            "calculate_sulfur_amount",
            {'gold_mass_g': gold_mass_g, 'equivalents': equivalents}
        )

@weave.op()
@safe_execute
@validate_input({
    'gold_mass_g': lambda x: x > 0,
    'equivalents': lambda x: x > 0
})
def calculate_nabh4_amount(gold_mass_g: float, 
                         equivalents: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate the amount of sodium borohydride needed
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
        equivalents: Number of molar equivalents (uses config default if None)
    
    Returns:
        dict: Contains calculated values with units
    
    Raises:
        CalculationError: If calculation fails
    """
    if equivalents is None:
        equivalents = config.DEFAULT_EQUIVALENTS_NABH4
    
    try:
        moles_gold = calculate_moles(gold_mass_g, config.MW_HAuCl4_3H2O)
        moles_nabh4 = moles_gold * equivalents
        mass_nabh4 = calculate_mass(moles_nabh4, config.MW_NaBH4)
        
        # Calculate solution volume (0.5 M in water)
        concentration_M = 0.5
        volume_solution_ml = (moles_nabh4 / concentration_M) * 1000
        
        result = {
            'moles_gold': round(moles_gold, 6),
            'moles_nabh4': round(moles_nabh4, 6),
            'mass_nabh4_g': round(mass_nabh4, 4),
            'volume_solution_ml': round(volume_solution_ml, 1),
            'concentration_M': concentration_M,
            'equivalents': equivalents,
            'units': {
                'moles_gold': 'mol',
                'moles_nabh4': 'mol',
                'mass_nabh4_g': 'g',
                'volume_solution_ml': 'mL',
                'concentration_M': 'M'
            }
        }
        
        # Log calculation
        safe_wandb_log({
            'calculation': {
                'type': 'nabh4_amount',
                'input': gold_mass_g,
                'output': mass_nabh4,
                'equivalents': equivalents
            }
        })
        
        return result
        
    except Exception as e:
        raise CalculationError(
            f"Failed to calculate NaBH4 amount: {str(e)}",
            "calculate_nabh4_amount",
            {'gold_mass_g': gold_mass_g, 'equivalents': equivalents}
        )

@weave.op()
@safe_execute
@validate_input({
    'gold_mass_g': lambda x: x > 0,
    'actual_yield_g': lambda x: x >= 0
})
def calculate_percent_yield(gold_mass_g: float, actual_yield_g: float) -> Dict[str, float]:
    """
    Calculate percent yield based on gold content
    
    Args:
        gold_mass_g: Mass of starting HAuCl4·3H2O in grams
        actual_yield_g: Mass of final Au25 nanoparticles in grams
    
    Returns:
        dict: Contains yield calculations with analysis
    
    Raises:
        CalculationError: If calculation fails
    """
    try:
        # Calculate gold content in starting material
        gold_fraction = config.MW_Au / config.MW_HAuCl4_3H2O
        theoretical_gold_mass = gold_mass_g * gold_fraction
        
        # For Au25 clusters, account for ligands (~30% by mass)
        ligand_fraction = 0.3
        theoretical_yield = theoretical_gold_mass / (1 - ligand_fraction)
        
        # Calculate percent yield
        if theoretical_yield > 0:
            percent_yield = (actual_yield_g / theoretical_yield) * 100
        else:
            percent_yield = 0
        
        # Determine yield quality
        if percent_yield >= config.TARGET_YIELD_PERCENT:
            yield_quality = "Excellent"
        elif percent_yield >= config.MIN_YIELD_PERCENT:
            yield_quality = "Good"
        else:
            yield_quality = "Poor"
        
        result = {
            'starting_mass_g': round(gold_mass_g, 4),
            'gold_content_g': round(theoretical_gold_mass, 4),
            'theoretical_yield_g': round(theoretical_yield, 4),
            'actual_yield_g': round(actual_yield_g, 4),
            'percent_yield': round(percent_yield, 2),
            'yield_quality': yield_quality,
            'target_yield_percent': config.TARGET_YIELD_PERCENT,
            'units': {
                'masses': 'g',
                'percent_yield': '%'
            }
        }
        
        # Log calculation
        safe_wandb_log({
            'calculation': {
                'type': 'percent_yield',
                'input': gold_mass_g,
                'actual_yield': actual_yield_g,
                'percent_yield': percent_yield,
                'quality': yield_quality
            }
        })
        
        return result
        
    except Exception as e:
        raise CalculationError(
            f"Failed to calculate percent yield: {str(e)}",
            "calculate_percent_yield",
            {'gold_mass_g': gold_mass_g, 'actual_yield_g': actual_yield_g}
        )

@weave.op()
@safe_execute
@validate_input({
    'gold_mass_g': lambda x: x > 0,
    'toab_mass_g': lambda x: x > 0
})
def calculate_toab_ratio(gold_mass_g: float, toab_mass_g: float) -> Dict[str, Union[float, str]]:
    """
    Calculate the molar ratio of TOAB to gold
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
        toab_mass_g: Mass of TOAB in grams
    
    Returns:
        dict: Contains ratio calculations and recommendations
    
    Raises:
        CalculationError: If calculation fails
    """
    try:
        moles_gold = calculate_moles(gold_mass_g, config.MW_HAuCl4_3H2O)
        moles_toab = calculate_moles(toab_mass_g, config.MW_TOAB)
        
        if moles_gold > 0:
            ratio = moles_toab / moles_gold
        else:
            ratio = 0
        
        # Determine if ratio is optimal (typically 1.5-2.5:1)
        if 1.5 <= ratio <= 2.5:
            ratio_assessment = "Optimal"
        elif ratio < 1.5:
            ratio_assessment = "Too low - may affect phase transfer"
        else:
            ratio_assessment = "Too high - excess TOAB may interfere"
        
        result = {
            'moles_gold': round(moles_gold, 6),
            'moles_toab': round(moles_toab, 6),
            'toab_to_gold_ratio': round(ratio, 2),
            'ratio_assessment': ratio_assessment,
            'recommended_ratio': '2:1',
            'units': {
                'moles': 'mol',
                'ratio': 'mol/mol'
            }
        }
        
        # Log calculation
        safe_wandb_log({
            'calculation': {
                'type': 'toab_ratio',
                'ratio': ratio,
                'assessment': ratio_assessment
            }
        })
        
        return result
        
    except Exception as e:
        raise CalculationError(
            f"Failed to calculate TOAB ratio: {str(e)}",
            "calculate_toab_ratio",
            {'gold_mass_g': gold_mass_g, 'toab_mass_g': toab_mass_g}
        )

@weave.op()
def calculate_all_reagents(gold_mass_g: float) -> Dict[str, Dict[str, float]]:
    """
    Calculate all reagent amounts for a complete synthesis
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
    
    Returns:
        dict: Complete reagent calculations
    """
    try:
        # Calculate all reagents
        sulfur_calc = calculate_sulfur_amount(gold_mass_g)
        nabh4_calc = calculate_nabh4_amount(gold_mass_g)
        
        # Calculate TOAB based on typical ratio
        moles_gold = calculate_moles(gold_mass_g, config.MW_HAuCl4_3H2O)
        toab_mass = calculate_mass(moles_gold * 2, config.MW_TOAB)  # 2:1 ratio
        
        # Calculate solvent volumes
        toluene_volume = config.DEFAULT_SOLVENT_VOLUME_ML
        
        result = {
            'gold': {
                'mass_g': round(gold_mass_g, 4),
                'moles': round(moles_gold, 6)
            },
            'toab': {
                'mass_g': round(toab_mass, 4),
                'moles': round(moles_gold * 2, 6)
            },
            'sulfur': sulfur_calc,
            'nabh4': nabh4_calc,
            'solvents': {
                'toluene_ml': toluene_volume,
                'water_ml': 50  # For NaBH4 solution
            },
            'total_volume_ml': toluene_volume + 50
        }
        
        return result
        
    except Exception as e:
        raise CalculationError(
            f"Failed to calculate all reagents: {str(e)}",
            "calculate_all_reagents",
            {'gold_mass_g': gold_mass_g}
        )