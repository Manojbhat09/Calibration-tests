
def rigid_transform(A, B, scaling):
	assert len(A) == len(B)

	N = A.shape[0]

	# Bring points to 0
	centroid_A = np.mean(A, axis =0)
	centroid_B = np.mean(B, axis=0)

	AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # Find the cross covariance matrix
    if scaling:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    # Find optimal direction (LS is complex)
    U, S, Vt = np.linalg.svd(H)

    # Find rotation to that direction
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scaling:
    	# Find variance
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))
        # Total translation
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t

# Testing 

if __name__=="__main__":

	A = np.matrix([[10.0,10.0,10.0],
	               [20.0,10.0,10.0],
	               [20.0,10.0,15.0]])

	B = np.matrix([[18.8106,17.6222,12.8169],
	               [28.6581,19.3591,12.8173],
	               [28.9554, 17.6748, 17.5159]])

	n = B.shape[0]

	Ttarg = np.matrix([[0.9848, 0.1737,0.0000,-11.5859],
	                   [-0.1632,0.9254,0.3420, -7.621],
	                   [0.0594,-0.3369,0.9400,2.7755],
	                   [0.0000, 0.0000,0.0000,1.0000]])

	Tstarg = np.matrix([[0.9848, 0.1737,0.0000,-11.5865],
	                   [-0.1632,0.9254,0.3420, -7.621],
	                   [0.0594,-0.3369,0.9400,2.7752],
	                   [0.0000, 0.0000,0.0000,1.0000]])

	scaling = 0 
	# recover the transformation
	s, ret_R, ret_t = rigid_transform(A, B, scaling)

	# Find the error
	B2 = (ret_R * B.T) + np.tile(ret_t, (1, n))
	B2 = B2.T
	err = A - B2
	err = np.multiply(err, err)
	err = np.sum(err)
	rmse = sqrt(err / n);

	#convert to 4x4 transform
	match_target = np.zeros((4,4))
	match_target[:3,:3] = ret_R
	match_target[0,3] = ret_t[0]
	match_target[1,3] = ret_t[1]
	match_target[2,3] = ret_t[2]
	match_target[3,3] = 1

	if scaling:
	    print("Total Diff to Tgt matrix")
	    print(np.sum(match_target - Tstarg))
	else:
	    print("Total Diff to Tgt matrix")
	    print(np.sum(match_target - Ttarg))

	print("RMSE", rmse)