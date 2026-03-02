#!/usr/bin/env python
"""
Test script to verify magnetic declination calculation.
"""
from datetime import datetime, date
import geomag

# Test location: Example coordinates (modify as needed)
# Using Brisbane, Australia as an example
lat = -27.4705
lon = 153.0260
altitude_m = 10.0
test_date = date(2026, 1, 8)  # Use date instead of datetime

# Calculate magnetic declination
altitude_km = altitude_m / 1000.0
declination = geomag.declination(lat, lon, altitude_km, test_date)

print(f"Test Magnetic Declination Calculation")
print(f"======================================")
print(f"Location: Lat={lat}°, Lon={lon}°")
print(f"Altitude: {altitude_m} m")
print(f"Date: {test_date}")
print(f"Magnetic Declination: {declination:.2f}°")
print()
print(f"Interpretation:")
if declination > 0:
    print(f"  Magnetic North is {abs(declination):.2f}° EAST of True North")
    print(f"  To convert magnetic heading to true heading: add {declination:.2f}°")
else:
    print(f"  Magnetic North is {abs(declination):.2f}° WEST of True North")
    print(f"  To convert magnetic heading to true heading: subtract {abs(declination):.2f}°")
print()
print(f"Example:")
print(f"  If gimbal yaw (magnetic) = 90°")
print(f"  True heading = 90° + {declination:.2f}° = {90 + declination:.2f}°")
