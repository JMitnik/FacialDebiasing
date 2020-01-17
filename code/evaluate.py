"""
In this file the evaluation of the network is done
"""

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", DEVICE)

def eval_model(model, data_loaders):
    """
    perform evaluation of a single epoch
    """

    face_loader, nonface_loader = data_loaders

    model.eval()
    avg_loss = 0
    avg_acc = 0

    all_labels = torch.LongTensor([]).to(DEVICE)
    all_preds = torch.Tensor([]).to(DEVICE)
    all_idxs = torch.LongTensor([]).to(DEVICE)

    with torch.no_grad():
        for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
            images, labels, idxs = concat_batches(face_batch, nonface_batch)
            batch_size = labels.size(0)

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            idxs = idxs.to(DEVICE)
            pred, loss = model.forward(images, labels)

            loss = loss/batch_size
            acc = calculate_accuracy(labels, pred)

            avg_loss += loss.item()
            avg_acc += acc

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_idxs = torch.cat((all_idxs, idxs))


    best_faces, worst_faces, best_other, worst_other = get_best_and_worst(all_labels, all_preds)
    visualize_best_and_worst(data_loaders, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other)

    return avg_loss/(i+1), avg_acc/(i+1)

def main():
    train_loaders: DataLoaderTuple
    valid_loaders: DataLoaderTuple

    train_loaders, valid_loaders = train_and_valid_loaders(
        batch_size=ARGS.batch_size,
        train_size=0.8,
        max_images=ARGS.dataset_size
    )

    # Load model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    eval_model()

    for epoch in range(ARGS.epochs):
        # Generic sequential dataloader to sample histogram

        print("Starting epoch:{}/{}".format(epoch, ARGS.epochs))

        if ARGS.debias_type != 'none':
            hist_loader = make_hist_loader(train_loaders.faces.dataset, ARGS.batch_size)
            hist = update_histogram(model, hist_loader, epoch)

            train_loaders.faces.sampler.weights = hist

        train_loss, train_acc = train_epoch(model, train_loaders, optimizer)
        print("training done")
        val_loss, val_acc = eval_epoch(model, valid_loaders, epoch)

        print("epoch {}/{}, train_loss={:.2f}, train_acc={:.2f}, val_loss={:.2f}, val_acc={:.2f}".format(epoch+1,
                                    ARGS.epochs, train_loss, train_acc, val_loss, val_acc))

        valid_data = concat_datasets(valid_loaders.faces.dataset, valid_loaders.nonfaces.dataset, proportion_a=0.5)
        print(valid_data)
        print_reconstruction(model, valid_data, epoch)

        with open("results/"+FOLDER_NAME + "/training_results.csv", "a") as write_file:
            s = "{},{},{},{},{}\n".format(epoch, train_loss, val_loss, train_acc, val_acc)
            print("S:", s)
            write_file.write(s)

        torch.save(model.state_dict(), "results/"+FOLDER_NAME+"/model.pt".format(epoch))

    return 

if __name__ == "__main__":
    print("start evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument("--adress", required=True, type=str)

    ARGS = parser.parse_args()

    main()