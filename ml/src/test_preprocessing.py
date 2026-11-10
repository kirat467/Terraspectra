
import numpy as np
from preprocessing import validate_hyperspectral_cube


# Test 1: Valid hyperspectral image
cube = np.random.rand(20, 20, 200)
result = validate_hyperspectral_cube(cube)

assert result.shape == (20, 20, 200)
print("Test 1 passed: Valid image")


# Test 2: Wrong dimensions
try:
    validate_hyperspectral_cube(np.random.rand(20, 200))
except ValueError:
    print("Test 2 passed: Wrong dimensions detected")


# Test 3: Empty image
try:
    validate_hyperspectral_cube(np.empty((0, 20, 200)))
except ValueError:
    print("Test 3 passed: Empty image detected")


# Test 4: Missing values
bad_cube = np.random.rand(20, 20, 200)
bad_cube[0, 0, 0] = np.nan

try:
    validate_hyperspectral_cube(bad_cube)
except ValueError:
    print("Test 4 passed: Missing values detected")


print("\nAll preprocessing tests completed!")
