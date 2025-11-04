import math
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CD

"""Calculate mass flow rate - Non-Homogeneous Non-Equilibrium Model"""
def m_ox_NHNE(Pres_Injector_bar, Pres_Chamber_bar, Temperature_Celcius, m_ox_1):
    # Constants
    L = 0.01  # Length (m)
    D = 0.0015  # Diameter (m)
    mu = 6.8e-5  # Dynamic Viscosity (Pa·s)
    Pt = Pres_Injector_bar * 1e5  # Convert bar to Pascal
    Pc = Pres_Chamber_bar * 1e5  # Convert bar to Pascal
    Temp_K = Temperature_Celcius + 273.15  # Convert Celsius to Kelvin 
    Kd = 2.28  # Additional losses constant 

    # Calculated values
    A_hole = math.pi * (D/2)**2  # Hole cross-section area (m^2)
    A_inj = numb_hole * A_hole  # Total injection area (m^2)
    rho_1 = CD.PropsSI('D', 'P', Pt, 'Q', 0, 'N2O') # Tank density (kg/m^3)
    rho_2 = CD.PropsSI('D', 'T', Temp_K, 'P', Pc, 'N2O')  # Chamber density (kg/m^3)
    h1 = CD.PropsSI('H', 'T', Temp_K, 'P', Pt, 'N2O')  # J/kg
    Pv = CD.PropsSI('P', 'T', Temp_K, 'Q', 0, 'N2O')  # Get saturation pressure (vapor pressure) in Pa    

    # Specific enthalpy in the chamber
    s1 = CD.PropsSI('S', 'T', Temp_K, 'Q', 0, 'N2O')  # J/kg-K
    sL = CD.PropsSI('S', 'P', Pc, 'Q', 0, 'N2O')  # J/kg-K
    sV = CD.PropsSI('S', 'P', Pc, 'Q', 1, 'N2O')  # J/kg-K 
    x = (s1-sL)/(sV-sL) # Quality
    print(f"Titulo={x}")
    hV = CD.PropsSI('H', 'P', Pc, 'Q', 1, 'N2O')  # J/kg
    hL = CD.PropsSI('H', 'P', Pc, 'Q', 0, 'N2O')  # J/kg
    h2 = (1-x)*hL + x*hV  # J/kg

    # Reynolds number calculation
    Re = (m_ox_1 * D) / (A_inj * mu)
    fD = 0.316 * Re**(-0.25)
    Cd = (1 / (fD * L/D/numb_hole + Kd))**0.5
    k = ((Pt-Pc)/(Pv-Pc))**0.5
    m_inc = (2*rho_1*(Pt-Pc))**0.5
    m_hem = rho_2*(2*(h1-h2))**0.5
    m = Cd * A_inj * ((1/(1+k))*m_inc+((k/(1+k))*m_hem))
    print(f"Cd={Cd}")
    return m

if __name__ == "__main__":
    while True:
        # Fixed Definitions
        P_tank = 44  # Tank pressure (bar)
        P_chamber_min = 1  # Minimum chamber pressure (bar)
        P_chamber_max = P_tank  # Maximum chamber pressure (bar)
        num_pontos = 50  # Number of points for chamber pressure variation
        Temperature = 10  # Temperature (Celsius)
        
        # User inputs for hole number range
        numb_hole_start = int(input("Enter the starting number of holes: "))
        numb_hole_end = int(input("Enter the ending number of holes: "))
        
        if numb_hole_start > numb_hole_end:
            print("Error: Starting number of holes cannot be greater than the ending number. Please try again.")
            continue
        
        numb_hole_step = int(input("Enter the increment for the number of holes: "))
        
        numb_holes_values = range(numb_hole_start, numb_hole_end + 1, numb_hole_step)
        
        if len(numb_holes_values) > 8:
            print("Error: Too many data points for the plot. Please try again.")
            continue  # Restart the loop for user input
        
        P_chamber = np.linspace(P_chamber_min, P_chamber_max, num_pontos)  # Chamber pressure variation
        
        # Create the figure
        plt.figure(figsize=(12, 6))
        
        for numb_hole in numb_holes_values:
            mass_flows = []
            for Pc in P_chamber:
                m = m_ox_NHNE(P_tank, Pc, Temperature, numb_hole)
                mass_flows.append(m)
            
            plt.plot(P_chamber, mass_flows, label=f"N = {numb_hole}")
        
        # Graph settings
        plt.title("Mass Flow Rate vs Chamber Pressure (D = 1,5mm)")
        plt.xlabel("Chamber Pressure [bar]")
        plt.ylabel("Mass Flow Rate [kg/s]")
        plt.legend(loc="upper left")
        plt.grid(True)
        
        # Add a small text box with the injector pressure value
        plt.text(0.95, 0.05, f"Injector Pressure: {P_tank} bar", horizontalalignment='right', 
                 verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=12, color="red")
        
        # Print the configurations
        print("\nUsed Configurations:")
        print(f"Tank Pressure: {P_tank} bar")
        print(f"Chamber Pressure: {P_chamber_min} to {P_chamber_max} bar")
        print(f"Temperature: {Temperature} °C")
        print(f"Hole number range from {numb_hole_start} to {numb_hole_end} with step {numb_hole_step}")

        # Display the plot only if the inputs are valid
        plt.show()
        break