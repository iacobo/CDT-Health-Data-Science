from itertools import permutations, combinations_with_replacement

###########################################
## Proof of concept
###########################################

# Testing whether actual data does coe in varied orientations:

for p in set(pid):
    mask_p = np.isin(pid, [p])
    x_p, y_p = X_raw[mask_p], y[mask_p]
    
    print(f'\nPatient ID: {p}')
    print(f'Orientation of x,y,z acelerometers: ', np.sign(np.mean(x_p[:,0])), np.sign(np.mean(x_p[:,1])), np.sign(np.mean(x_p[:,2])))
    
# Does: thus, augmentation useful
    
#####################################################

from collections import defaultdict

X_raw_full = np.load('X_raw.npy', mmap_mode='r')
pid_full = data['pid']

dicty = defaultdict(lambda: 0)

for p in set(pid_full):
    mask_p = np.isin(pid_full, [p])
    x_p, y_p = utils.ArrayFromMask(X_raw_full, mask_p), data['y'][mask_p]
    
    try:
        #print(f'\nPatient ID: {p}')
        orients = (np.sign(np.mean(x_p[:][0])), np.sign(np.mean(x_p[:][1])), np.sign(np.mean(x_p[:][2])))
        #print(f'Orientation of x,y,z acelerometers: ', orients)
        dicty[orients] += 1
    except:
        print(f'\nPatient ID: {p}')

print(dicty)

# ( 1.0,  1.0,  1.0): 72
# (-1.0, -1.0, -1.0): 57 
# (-1.0,  1.0,  1.0):  3
# ( 1.0,  1.0, -1.0):  3
# ( 1.0, -1.0, -1.0):  2
# (-1.0,  1.0, -1.0):  1

#####################################################


import matplotlib.pyplot as plt

# Plotting histograms of per patient average values for each coordinate acceleration for each activity
# stratified into groups of 'average' coordinate value 3-tuples.

for coord in (0,1,2):

    fig, ax = plt.subplots(nrows=len(set(data['y'])), ncols=len(dicty.keys()))

    for i, activity in enumerate(set(data['y'])):
        act_mask = np.isin(data['y'], activity)
        for j, orient in enumerate(dicty.keys()):
            print('Average patient orientation: ', orient)
            print('Activity: ', activity)
            
            ids = dicty2[orient]
            id_mask = np.isin(pid_full, ids)
            full_mask = id_mask & act_mask
            x = utils.ArrayFromMask(X_raw_full, full_mask)
            print(x.shape)
            
            # Plot histograms of orientation type vs activity
            x_r = []
            for patient in x:
                m = np.mean(patient[coord])
                x_r.append(m)
            ax[i,j].hist(x_r, label={0:'x',1:'y',2:'z'}[coord], alpha=0.5)
    plt.show()

    
#####################################################
#####################################################

###############################################
# Augmented cases:
###############################################
# - Simulate left- vs right-handedness
# - Simulate different orientations of device in housing (normal, rotated x, flipped y, flip y + rot x)   

print("\nAugmenting dataset by simulating all 8 possible flipped/rotated/handedness orientations of device.")
print("\nExtracting features on pseudo-training set...")

flips = [x for flips in combinations_with_replacement([1,-1],3) for x in set(permutations(flips))]
flips.remove((1,1,1))

# Check data hasn't already been augmented
if X_train.shape == X_feats[mask_train].shape:
    for flip in flips:
        X_train_aug = np.empty_like(X_train)
        for i in tqdm(range(X_raw_train.shape[0])):
            x = X_raw_train[i].copy()
            x[0,:] *= flip[0]
            x[1,:] *= flip[1]
            x[2,:] *= flip[2]
            X_train_aug[i] = extractor.extract(x)
        if not X_train_out:
            X_train_out = np.concatenate((X_train, X_train_aug))
        else:
            X_train_out = np.concatenate((X_train_out, X_train_aug))

        # Add in the "new data" to training set
        y_train_out = np.concatenate((y_train_out, y_train))

# Add in the "new data" to training set
#y_train = np.concatenate((y_train) * len(flips)+1)
print(f"Shape of new augmented X_train: {X_train_out.shape}")
