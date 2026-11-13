### Copy and paste the code from your designed filter function here in this file. ###
from scipy import signal
import numpy as np
from numpy.typing import NDArray


def print_transfer_function(b: NDArray, a: NDArray) -> None:
    """
    Print the transfer function in a readable format: H(z) = numerator/denominator
    
    Arguments:
    ----------
        b: Numerator coefficients
        a: Denominator coefficients
    """
    # Round coefficients to 4 decimal places for cleaner display
    b = np.round(b, 4)
    a = np.round(a, 4)
    
    def coeffs_to_string(coeffs: NDArray, var: str = 'z') -> str:
        terms = []
        for i, coeff in enumerate(coeffs):
            if coeff == 0:
                continue
            # Determine the exponent
            exp = len(coeffs) - i - 1
            
            # Format the coefficient
            if abs(coeff) == 1 and exp > 0:
                coeff_str = '-' if coeff < 0 else ''
            else:
                coeff_str = f"{coeff:g}"
            
            # Format the term
            if exp == 0:
                terms.append(coeff_str)
            elif exp == 1:
                terms.append(f"{coeff_str}{var}")
            else:
                terms.append(f"{coeff_str}{var}^{exp}")

        result = " + ".join(terms) if terms else "0" 
        return result.replace(" + -", " - ")
    
    numerator = coeffs_to_string(b)
    denominator = coeffs_to_string(a)
    
    print(f"H(z) = ({numerator}) / ({denominator})")

def main() -> None:
    sos = design_filter()
    b, a = signal.sos2tf(sos)
    print_transfer_function(b, a)

if __name__ == "__main__":
    main()