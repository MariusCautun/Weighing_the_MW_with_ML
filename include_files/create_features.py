import numpy as np
import h5py as h5


def load_data_and_create_features( inputFile ):
    """Function for loading the data for the galaxies and their satellite systems used to constrain the mass of the Milky Way by training various ML algorithms. This function also manipulates some of the feature by transforming them to log-space.
    
    INPUT:
            inputFile - the name of the data input file
    
    OUTPUT:
            data_input - a (N x m) numpy array giving the input 'm' features for the 'N' 
                        entries in the file
            data_output - a (N x 1) numpy array giving the target output for each entry
            name_input - the list of names for each feature in the input data
            name_output - the name of the output variable
    """
    with h5.File( inputFile, 'r' ) as hf:
        mw_Mhalo = np.array( hf["M_halo"] )
        mw_Mstar = np.array( hf["M_star"] )
        mw_lum_func = np.array( hf["luminosity_function"] )
        mw_vel_dis  = np.array( hf["velocity_dispersion"] )
        mw_vel_rad  = np.array( hf["velocity_dispersion_radial"] )
        mw_ang_mom  = np.array( hf["mean_angular_momentum"] )
        mw_dis      = np.array( hf["mean_distance"] )

    num_systems = mw_Mhalo.shape[0]
    num_features = 1 + mw_lum_func.shape[1] + 1*4

    print( "The '%s' input file contains:" % inputFile )
    print( "MW-analogues:   %i" % mw_Mhalo.shape[0] )


    # transform the masses, velocity dispersion and angular momentum to log values
    mw_Mhalo =  np.log10(mw_Mhalo)
    
    mw_Mstar =  np.log10(mw_Mstar)
    mw_Mstar[mw_Mstar<8.] = 8.    # get rid of the tail -- very few entries have these values
    
    # calculate the PDF of the luminosity function
    # (the file contains the CDF)
    for i in range(mw_lum_func.shape[1]-1):
        mw_lum_func[:,i] -= mw_lum_func[:,i+1]
    
    # calculate the tangetial velocity dispersion
    mw_vel_dis -= mw_vel_rad
    
    sel = mw_vel_dis > 100.  # all the systems with valid values
    mw_vel_dis[ sel] = np.log10(mw_vel_dis[sel])
    mw_vel_dis[~sel] = np.log10(100.)

    sel = mw_vel_rad > 100.  # all the systems with valid values
    mw_vel_rad[ sel] = np.log10(mw_vel_rad[sel])
    mw_vel_rad[~sel] = np.log10(100.)

    sel = mw_ang_mom > 1  # all the systems with valid values
    mw_ang_mom[ sel] = np.log10(mw_ang_mom[sel])
    mw_ang_mom[~sel] = 0.


    # define lists storing the name of each feature
    name_input = [ "M_star", "N_sat 1.e6", "N_sat 1.e7", "N_sat 1.e8", "N_sat 1.e9",  "N_sat 1.e10", 
                   "vel. tan.", "vel. radial", "mean L", "mean d"
                 ]
    name_output = [ "M_halo" ]

    # merge the input and output features in two different array
    data_input = np.column_stack( ( mw_Mstar,mw_lum_func, mw_vel_dis, mw_vel_rad, mw_ang_mom, mw_dis) )
    data_output= mw_Mhalo.reshape(-1,1)
    
    return data_input, data_output, name_input, name_output 