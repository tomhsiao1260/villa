import docker
import os
import yaml
import re
from tqdm import tqdm
from copy import deepcopy

class ScrollBuilder():
    """
    Building the derived scroll data formats from the raw data formats.
    """
    def __init__(self, config_file='builder.yaml'):
        self.client = docker.from_env()
        self.config_file = config_file
        self.config = self.load_config()
        self.base_path = self.config.get('base_path', os.getcwd())

    def load_config(self):
        # Load the builder file
        with open(self.config_file, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
        
    def build_regex(self, variables, input_patterns={}):
        # parses the configuration arguments to valid regex patterns
        def replace_regex(regex, patterns):
            # Replace any placeholders with the completed regex patterns of earlier variables
            for name, pattern in patterns.items():
                regex = regex.replace("${" + name + "}", pattern)
            return regex
        
        completed_patterns = {}
        for key, regex in variables.items():
            # Replace any placeholders with the completed regex patterns of earlier variables
            regex = replace_regex(regex, input_patterns)
            regex = replace_regex(regex, completed_patterns)
            # Encapsulate in named groups to reference directly
            completed_patterns[key] = f"(?P<{key}>{regex})"
        return completed_patterns
    
    def fill_regex(self, regex, previous_match_result):
        """
        Fill a compiled regex pattern using previous regex match results.

        Parameters:
            regex: The regex pattern to use for matching.
            previous_match_result (re.Match): A regex match object containing previously captured groups.

        Returns:
            str: The filled regex pattern.
        """
        # Construct the regex dynamically based on previous matches
        filled_regex = regex
        for name in previous_match_result.re.groupindex.keys():
            # Use re.escape to safely insert the previous match into the regex
            filled_regex = filled_regex.replace(f"(?P<{name}>.*?)", re.escape(previous_match_result.group(name)))

        return filled_regex
    
    def search_filesystem(self, regex_pattern):
        # Search the filesystem for files matching the regex pattern
        matches = set()
        compiled_regex = re.compile(regex_pattern)
        for dirpath, dirnames, filenames in tqdm(os.walk(self.base_path)):
            # remove base part of path
            dirpath = dirpath[len(self.base_path)+1:]
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                # print(dirpath, dirnames, filename, full_path)
                match = compiled_regex.search(full_path)
                if match:
                    match_dict = match.groupdict()
                    # delete key "permutation" from groupdict
                    del match_dict["permutation"]
                    matches.add(tuple(match_dict.items()))
        matches = [{key: value for key, value in match} for match in matches]
        return matches
    
    def match_permutations(self, permutations, base_patterns):
        # Recursively match all permutations
        if len(permutations) == 0:
            return [base_patterns]
        permutation = permutations[0]
        permutation_patterns = self.build_regex({"permutation": permutation}, base_patterns)
        permutation_pattern = permutation_patterns["permutation"]
        # Match permutation and update base patterns
        matches = self.search_filesystem(permutation_pattern)
        # Recurse on remaining permutations
        matched_patterns = []
        for match in matches:
            base_patterns_ = deepcopy(base_patterns)
            base_patterns_.update(match)
            matched_patterns += self.match_permutations(permutations[1:], base_patterns_)

        # Find unique matched patterns dicts
        matched_patterns = [dict(s) for s in set(frozenset(d.items()) for d in matched_patterns)]

        return matched_patterns

    def script_recomputation(self, script_config):
        # TODO: Check if the data changed since last computation
        return True
    
    def build_command(self, command_str, script_config):
        # Build the command, replace all variables with the matched patterns
        for key, value in script_config.items():
            command_str = command_str.replace("${" + key + "}", value)
        return command_str
    
    def build_commands(self, script, matched_patterns):
        # Builds docker and scrip commands
        script_configurations = []
        for matched_pattern in matched_patterns:
            script_configuration = {}
            commands = script['commands']
            for command in commands:
                docker_command = command['docker_command']
                for volume in docker_command['volumes']:
                    print(volume)
                    for key, value in volume.items():
                        volume[key] = self.build_command(value, matched_pattern)
                script_configuration['docker_command'] = docker_command
                script_commands = command['script_commands']
                script_commands_list = []
                for script_command in script_commands:
                    script_command = self.build_command(script_command, matched_pattern)
                    script_commands_list.append(script_command)
                script_configuration['script_commands'] = script_commands_list
            script_configurations.append(script_configuration) 

        # Unique script configurations
        return script_configurations
    
    def get_script_commands(self, script):
        # Finds all script path configurations
        base_patterns = self.build_regex(script['variables'])
        # Add base path to base patterns
        base_patterns['base_path'] = self.base_path
        # Returns all permutations of the script configurations for execution of the script
        matched_patterns = self.match_permutations(script['permutations'], base_patterns)
        print(f"Matched Patterns: {matched_patterns}")

        return self.build_commands(script, matched_patterns)
    
    def run_docker_container(self, docker_command):
        # Environment variables
        docker_envs = docker_command['environment']
        environment = {
            key: os.getenv(value, value) for key, value in docker_envs.items()
        }

        # Volumes
        docker_volumes = docker_command['volumes']
        volumes = {
            volume_dict['host_path']: {"bind": volume_dict['container_path'], "mode": "rw"}for volume_dict in docker_volumes
        }

        # Run the Docker container
        try:
            container = self.client.containers.run(
                docker_command['name'],
                "tail -f /dev/null",  # Keeps the container running for executing further commands
                runtime="nvidia",  # This is needed to utilize GPU
                shm_size='150g',  # Setting the shared memory size to 150GB
                environment=environment,
                volumes=volumes,
                detach=True,
                remove=True,  # Equivalent to --rm
                auto_remove=True  # Automatically remove the container when it exits
            )
            print("Container started successfully.")
            return container
        except Exception as e:
            print(f"Failed to start the container: {e}")
            return None

    def execute_script_inside_container(self, container, script_commands):
        try:
            for command in script_commands:
                print(f"Executing command inside container: {command}")
                exec_log = container.exec_run(command, workdir="/workspace")
                print("Output:", exec_log.output.decode())
        except Exception as e:
            print("Failed to execute script inside Docker:", e)

    def run_script(self, script_config):
        # Check if the script needs to be recomputed
        if not self.script_recomputation(script_config):
            return

        # Extract docker command and script commands from the configuration
        docker_command = script_config['docker_command']
        script_commands = script_config['script_commands']

        # Run the script
        print(f"Running Script: {script_config}")

        # Start Docker container
        container = self.run_docker_container(docker_command)
        if container:
            # Execute scripts inside container
            self.execute_script_inside_container(container, script_commands)
    
    def build(self):
        # Build all the output data formats
        for script in self.config['scripts']:
            script_commands = self.get_script_commands(self.config['scripts'][script])
            for script_config in tqdm(script_commands, desc=f"Building {script}"):
                self.run_script(script_config)

# Main
if __name__ == '__main__':
    builder = ScrollBuilder()
    builder.build()

# Example: python3 builder.py
# Stopp all containers: sudo docker rm -f $(sudo docker ps -aq)
