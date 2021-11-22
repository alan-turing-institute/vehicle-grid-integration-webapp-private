"""azureOptsXmpls.py
A file containing a list of example dict that could be sent by a user to
build a network and the run a simulation.

Possible use cases are from the ICL whitepaper:
https://europe.nissannews.com/en-GB/releases/release-beb6420a9f2916baf13a4ed15006b51d-nissan-eon-drive-and-imperial-college-highlight-the-carbon-saving-and-economic-benefits-of-vehicle-to-grid-technology
- 'energy-like': DA trading, 3rd part cost avoidance, DNO flex services
- 'power-like' (response in seconds): Dynamic containment, FFR
Not considered: Triad avoidance as this is expiring in a few years.
"""

# Dict values of None not yet implemented or simple possibility
run_dict0 = {
        # First: how to construct the MV-LV networks
        'network_data':{
            'n_id':1060, # MV-LV circuit ID, at the moment either 1060 or 1061
            'xfmr_scale':None, # Scale the transformer
            'mod_inc':None, # percentage i & c customers (i.e., day charging)
            'lv_sel':'n_lv', # method of building the MV/LV circuit
            'n_lv':5, # number of LV circuits (integer >=1 ) [lv_sel]
            'lv_ilist':[0,10,30,], # select the ith row of the LV ckts [lv_sel] 
            'lv_list':['1101','1141','1164',], # exact lv network ids [lv_sel]
        },
     
        # Next: allocating loads and generators. See help(turingNet.set_ldsi)
        # for options; if any of these are set to None, then they will not
        # be allocated.
        'dmnd_gen_data':{
            'rs':   {'lv':'lv',
                     'mv':'evens',},
            'ic':   {'lv':None,
                     'mv':'odds',},
            'ovnt': {'lv':'rs',
                     'mv':'rs',},
            'dytm': {'lv':'ic',
                     'mv':'ic',},
            'slr':  {'lv':None,
                     'mv':None,},
            'hps':  {'lv':None,
                     'mv':None,},
        },
        'simulation_data':{
            # Time series profiles types. See help(turingNet.set_load_profiles
            # to see options here.
            'ts_profiles':{
                    'rs':   {'lv':'crest',
                             'mv':'crest_'},
                    'ic':   {'lv':None,
                             'mv':'ic00_prof',},
                    'ovnt': {'lv':'ee',
                             'mv':None,},
                    'dytm': {'lv':None,
                             'mv':'ee',},
                    'slr':  {'lv':None,
                             'mv':None,},
                    'hps':  {'lv':None,
                             'mv':None,},
                    'n':144 # resolution, samples per day
                },
            'sim_type':None, # type of simulation, e.g., da-ahead v2g, etc
        },
}