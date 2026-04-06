from torch import save

def save_checkpoint(epoch, model, optim, sched, loss, filepath,
                    show_msg=False):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "loss": loss
    }

    save(checkpoint, filepath)
    if show_msg:
        print(f"SAVED checkpoint to {str(filepath)}")
