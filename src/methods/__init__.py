# UDA method modules.
# Each module exposes:
#   setup(model, args, device) -> dict   # method-specific components (e.g. discriminator)
#   trainer(src_dataloaders, tgt_loader, model, optimizer, args, device,
#           best_modelname, components, tgt_val_loader=None, tgt_genre=None)
