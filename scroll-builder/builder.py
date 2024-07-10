import os
import yaml
from tqdm import tqdm

class ScrollBuilder():
    def __init__(self, config_file='builder.yaml'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
        
    def script_recomputation(self, script_config):
        return True
    
    def build_docker_command(self, script_config):
        # Build the docker command to run the script
        # TODO
        return ""
    
    def run_script(self, script_config):
        # Check if the script needs to be recomputed
        if not self.script_recomputation(script_config):
            return
        
        # Build the docker command
        command = self.build_docker_command(script_config)

        # Run the script
        # TODO: add everything to start the docker image and execute the script
    
    def get_script_configurations(self, script):
        # Returns all permutations of the script configurations for execution of the script
        # TODO
        return []
    
    def build(self):
        # Build all the scripts
        for script in self.config['scripts']:
            script_configurations = self.get_script_configurations(self.config['scripts'][script])
            for script_config in tqdm(script_configurations, desc=f"Building {script}"):
                self.run_script(script_config)

# Main
if __name__ == '__main__':
    builder = ScrollBuilder()
    builder.build()
        
