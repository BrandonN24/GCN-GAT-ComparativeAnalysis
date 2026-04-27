import training.GCN_training as GCN_training
import training.cora as cora
import training.citeseer as citeseer

def main():

    # Load the Cora dataset
    cora_data, cora_num_classes = cora.load_cora()

    # Load the Citeseer dataset
    citeseer_data, citeseer_num_classes = citeseer.load_citeseer()

    # Prompt user for model choice to train
    model_choice = input("Enter the model to train ( [1] GCN Two Layer / [2] GCN Three Layer / [3] GAT Two Layer / [4] GAT Three Layer): ")

    match model_choice:
        case '1':
            print("Training GCN Two Layer on Cora dataset...")
            GCN_training.GCN_two_layer_training(cora_data, num_classes=cora_num_classes, epochs=200, dataset_name='Cora')
            print("Training GCN Two Layer on Citeseer dataset...")
            GCN_training.GCN_two_layer_training(citeseer_data, num_classes=citeseer_num_classes, epochs=200, dataset_name='Citeseer')
        case '2':
            print("Training GCN Three Layer on Cora dataset...")
            GCN_training.GCN_three_layer_training(cora_data, num_classes=cora_num_classes, epochs=200, dataset_name='Cora')
            print("Training GCN Three Layer on Citeseer dataset...")
            GCN_training.GCN_three_layer_training(citeseer_data, num_classes=citeseer_num_classes, epochs=200, dataset_name='Citeseer')
        case '3':
            print("GAT Two Layer training not implemented yet.")
        case '4':
            print("GAT Three Layer training not implemented yet.")
        case _:
            print("Invalid choice. Please enter a number between 1 and 4.")

    return

if __name__ == "__main__":
    main()