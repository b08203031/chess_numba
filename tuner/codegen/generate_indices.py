
import tuner.parameters as parameters

def generate_indices():
    pm = parameters.ParameterManager()
    
    print("# Generated indices")
    
    for p in pm.param_map:
        name = p['name']
        start = p['start']
        print(f"IDX_{name} = {start}")

if __name__ == "__main__":
    generate_indices()
