"""
Chemistry calculation utilities for nanoparticle synthesis
"""

# Molecular weights (g/mol)
MW_HAuCl4_3H2O = 393.83  # Gold(III) chloride trihydrate
MW_PhCH2CH2SH = 138.23   # 2-Phenylethanethiol
MW_NaBH4 = 37.83        # Sodium borohydride
MW_TOAB = 546.78        # Tetraoctylammonium bromide
MW_Au = 196.97          # Gold atomic weight

def calculate_moles(mass_g, molecular_weight):
    """Calculate moles from mass and molecular weight"""
    return mass_g / molecular_weight

def calculate_mass(moles, molecular_weight):
    """Calculate mass from moles and molecular weight"""
    return moles * molecular_weight

def calculate_sulfur_amount(gold_mass_g, equivalents=3):
    """
    Calculate the amount of 2-phenylethanethiol needed
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
        equivalents: Number of molar equivalents (default 3)
    
    Returns:
        dict: Contains calculated values
    """
    moles_gold = calculate_moles(gold_mass_g, MW_HAuCl4_3H2O)
    moles_sulfur = moles_gold * equivalents
    mass_sulfur = calculate_mass(moles_sulfur, MW_PhCH2CH2SH)
    
    return {
        'moles_gold': moles_gold,
        'moles_sulfur': moles_sulfur,
        'mass_sulfur_g': mass_sulfur,
        'equivalents': equivalents
    }

def calculate_nabh4_amount(gold_mass_g, equivalents=10):
    """
    Calculate the amount of sodium borohydride needed
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
        equivalents: Number of molar equivalents (default 10)
    
    Returns:
        dict: Contains calculated values
    """
    moles_gold = calculate_moles(gold_mass_g, MW_HAuCl4_3H2O)
    moles_nabh4 = moles_gold * equivalents
    mass_nabh4 = calculate_mass(moles_nabh4, MW_NaBH4)
    
    return {
        'moles_gold': moles_gold,
        'moles_nabh4': moles_nabh4,
        'mass_nabh4_g': mass_nabh4,
        'equivalents': equivalents
    }

def calculate_percent_yield(gold_mass_g, actual_yield_g):
    """
    Calculate percent yield based on gold content
    
    Args:
        gold_mass_g: Mass of starting HAuCl4·3H2O in grams
        actual_yield_g: Mass of final Au25 nanoparticles in grams
    
    Returns:
        dict: Contains yield calculations
    """
    # Calculate gold content in starting material
    gold_fraction = MW_Au / MW_HAuCl4_3H2O
    theoretical_gold_mass = gold_mass_g * gold_fraction
    
    # For Au25 clusters, we assume the yield is based on gold content
    # This is simplified - actual stoichiometry is more complex
    theoretical_yield = theoretical_gold_mass
    
    # Calculate percent yield
    if theoretical_yield > 0:
        percent_yield = (actual_yield_g / theoretical_yield) * 100
    else:
        percent_yield = 0
    
    return {
        'starting_mass_g': gold_mass_g,
        'gold_content_g': theoretical_gold_mass,
        'theoretical_yield_g': theoretical_yield,
        'actual_yield_g': actual_yield_g,
        'percent_yield': percent_yield
    }

def calculate_toab_ratio(gold_mass_g, toab_mass_g):
    """
    Calculate the molar ratio of TOAB to gold
    
    Args:
        gold_mass_g: Mass of HAuCl4·3H2O in grams
        toab_mass_g: Mass of TOAB in grams
    
    Returns:
        dict: Contains ratio calculations
    """
    moles_gold = calculate_moles(gold_mass_g, MW_HAuCl4_3H2O)
    moles_toab = calculate_moles(toab_mass_g, MW_TOAB)
    
    if moles_gold > 0:
        ratio = moles_toab / moles_gold
    else:
        ratio = 0
    
    return {
        'moles_gold': moles_gold,
        'moles_toab': moles_toab,
        'toab_to_gold_ratio': ratio
    }