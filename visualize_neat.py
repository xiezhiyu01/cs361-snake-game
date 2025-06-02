import argparse
import pickle
import neat
import graphviz


def draw_net(config, genome, filename="neat_net"):
    dot = graphviz.Digraph(format="png")
    # node_names = set()

    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys

    # Traverse backwards from outputs to find relevant nodes
    relevant_nodes = set(output_keys)
    added = True
    while added:
        added = False
        for (in_node, out_node), conn in genome.connections.items():
            if conn.enabled and out_node in relevant_nodes and in_node not in relevant_nodes:
                relevant_nodes.add(in_node)
                added = True

    # Draw relevant nodes
    for node_id in relevant_nodes:
        if node_id in input_keys:
            idx = input_keys.index(node_id) + 1
            label = f"I{idx:02d}"
            color = "lightblue"
        elif node_id in output_keys:
            idx = output_keys.index(node_id) + 1
            label = f"O{idx}"
            color = "yellow"
        else:
            label = ""
            color = "lightgray"
        dot.node(str(node_id), label=label, style="filled", fillcolor=color)

    # Step 3: Draw relevant edges with capped arrow thickness
    for (in_node, out_node), conn in genome.connections.items():
        if conn.enabled and in_node in relevant_nodes and out_node in relevant_nodes:
            width = min(5.0, 0.5 + abs(conn.weight))  # Cap thickness
            dot.edge(str(in_node), str(out_node), penwidth=str(width), color="black")

    dot.render(filename, cleanup=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='checkpoints/exp3-best-copy.pkl', help="Path to NEAT genome pickle")
    parser.add_argument("--config_path", type=str, default='neat_config.txt', help="Path to NEAT config file")
    parser.add_argument("--output", type=str, default="neat_network", help="Output file prefix")
    args = parser.parse_args()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config_path
    )

    with open(args.model_path, "rb") as f:
        genome = pickle.load(f)

    draw_net(config, genome, args.output)

if __name__ == "__main__":
    main()
