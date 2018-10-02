import sys;

def get_p_at_n_in_m(data, n, m, ind):
	pos_score = data[ind][0];
	curr = data[ind:ind+m];
	curr = sorted(curr, key = lambda x:x[0], reverse=True)

	if curr[n-1][0] <= pos_score:
		return 1;
	return 0;

def evaluate(file_path):
	data = []
	with open(file_path, 'r') as file:
		for line in file:
			line = line.strip();
			tokens = line.split("\t")
		
			if len(tokens) != 2:
				continue
		
			data.append((float(tokens[0]), int(tokens[1])));
		
	#assert len(data) % 10 == 0
	
	p_at_1_in_2 = 0.0
	p_at_1_in_10 = 0.0
	p_at_2_in_10 = 0.0
	p_at_5_in_10 = 0.0

        length = len(data)/10

	for i in xrange(0, length):
		ind = i * 10
		assert data[ind][1] == 1
	
		p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
		p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
		p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
		p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

	return (p_at_1_in_2/length, p_at_1_in_10/length, p_at_2_in_10/length, p_at_5_in_10/length)
	

