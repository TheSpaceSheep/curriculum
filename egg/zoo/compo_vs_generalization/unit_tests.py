import torch
from egg.zoo.compo_vs_generalization.data import (
        one_hotify, 
        ScaledDataset, 
        build_datasets, 
        mask_attributes
    )

def test_mask_attributes():
    batch_size = 10
    n_attributes = 5
    n_values = 3
    n_masked_attributes = 2
    
    # create dummy dataset
    train, _, _ = \
        build_datasets(n_attributes, 
                       n_values,
                       100,
                       10,
                       10,
                       rng=torch.Generator())

    train = one_hotify(train, n_attributes, n_values)

    train = ScaledDataset(train, 1)
    train = torch.utils.data.DataLoader(train, batch_size=batch_size)

    sender_input = next(iter(train))[0]

    # create a batch of random indices
    idx = []
    for _ in range(batch_size):
        idx.append(torch.randperm(n_attributes)[:n_masked_attributes])
    idx = torch.stack(idx)

    # mask input
    masked_input = mask_attributes(sender_input, idx, n_attributes, n_values) 

    # mask input and remove masked data
    hard_masked_input = mask_attributes(sender_input, 
            idx, 
            n_attributes,
            n_values,
            remove_masked_data=True
        ) 

    # mask input by last value
    masked_with_last_value_input = mask_attributes(sender_input_,
        idx,
        n_attributes,
        n_values+1,
        mask_by_last_value=True
    )

    print(idx.shape)
    print(idx)
    print(masked_input.shape)
    print(masked_input)
    print(hard_masked_input.shape)
    print(hard_masked_input)
    print(masked_with_last_value.shape)
    print(masked_with_last_value)

    assert idx.shape == torch.Size([batch_size, n_masked_attributes])
    assert masked_input.shape == torch.Size([batch_size, n_attributes*n_values])
    assert hard_masked_input.shape == torch.Size([batch_size, (n_attributes-n_masked_attributes)*n_values])
    assert (masked_with_last_value_input[0, idx[0][0]*(n_values+1):(idx[0][0]+1)*(n_values+1) ] == torch.tensor([0, 0, 0, 1])).all()


    # verify that the first index has been masked
    assert (masked_input[0, idx[0][0]*n_values:(idx[0][0]+1)*n_values ] == torch.tensor([0, 0, 0])).all()

    # same with input masked by last value
    assert (masked_with_last_value_input[0, idx[0][0]*(n_values+1):(idx[0][0]+1)*(n_values+1) ] == torch.tensor([0, 0, 0, 1])).all()


    print("--- Tests for mask_attributes successfully passed ---")


if __name__ == "__main__":
    test_mask_attributes()
