
"""
THIS FILE CONTAINS FUNCTIONS THAT CONTRIBUTE TO THE WRITING OF OUTPUTS
"""
import shutil
import os

def clean_results(nodes_central):
    
    shutil.rmtree('./output/'+str(nodes_central)+'_nodes/') 
    os.mkdir('./output/'+str(nodes_central)+'_nodes/')
    
    


