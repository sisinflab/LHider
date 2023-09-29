def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
    @return: None
    """
    for arg, value in arguments.items():
        print(f'{arg}: {value}')
