

def runRandomForests(t_value, examples):

    for i in range(0, t_value):
        # Get bootstrap example
        bootstrap_sample = get_bootstrap_sample(examples)






def get_bootstrap_sample(examples):
    bootstrap_samples = []
    for j in range(0, int(len(data_examples))):
        bootstrap_samples.append(examples[random.randint(0, len(examples) - 1)])
    return bootstrap_samples


def randTreeLearn():