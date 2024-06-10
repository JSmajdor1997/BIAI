from typing import Dict, Callable, Tuple, List

from genetics.Genome import Genome


class NeuralNetworkMap:
    layers: List[int]
    nr_of_inputs: int
    __neurons_indices_map: Dict[int, Dict[int, Tuple[int, int]]]

    def __init__(self, layers: List[int], nr_of_inputs: int):
        self.layers = layers
        self.nr_of_inputs = nr_of_inputs

        genome_index = 0

        neurons_indices_map: Dict[int, Dict[int, Tuple[int, int]]] = {}

        for layer_index in range(len(self.layers)):
            nr_of_inputs = self.nr_of_inputs if layer_index == 0 else self.layers[layer_index - 1]

            for neuron_index in range(self.layers[layer_index]):
                genome_weights_range = (genome_index, genome_index + nr_of_inputs)
                genome_index += nr_of_inputs

                if layer_index not in neurons_indices_map:
                    neurons_indices_map[layer_index] = {}

                neurons_indices_map[layer_index][neuron_index] = genome_weights_range

        self.__neurons_indices_map = neurons_indices_map

    def getWeights(self, genome: Genome, layer_index: int, neuron_index: int):
        indices = self.__neurons_indices_map[layer_index][neuron_index]

        return genome[indices[0]:indices[1]]

    def get_total_nr_of_genes(self) -> int:
        total_nr_of_genes = self.nr_of_inputs * self.layers[0]

        for layer_index in range(1, len(self.layers)):
            total_nr_of_genes += self.layers[layer_index - 1] * self.layers[layer_index]

        return total_nr_of_genes

    def getResults(
            self,
            genome: Genome,
            input_values: List[float],
            activation_function: Callable[[float], float],
    ) -> List[float]:
        if len(input_values) != self.nr_of_inputs:
            raise ValueError("Input values length must be equal to number of inputs of neural network")

        current_outputs: List[float] = []

        for neuron_index in range(self.layers[0]):
            weights = self.getWeights(genome, 0, neuron_index)
            value = activation_function(
                sum(input_value * weights[index] for index, input_value in enumerate(input_values))
            )
            current_outputs.append(value)

        for layer_index in range(1, len(self.layers)):
            previous_layer_outputs = current_outputs
            current_outputs = []

            for neuron_index in range(self.layers[layer_index]):
                weights = self.getWeights(genome, layer_index, neuron_index)
                value = activation_function(
                    sum(previous_layer_output * weights[index] for index, previous_layer_output in
                        enumerate(previous_layer_outputs))
                )
                current_outputs.append(value)

        return current_outputs
