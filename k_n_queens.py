import sympy
from sympy.logic.boolalg import Not, And
import numpy as np

class ChessBoard:
	def __init__(self, n, k=0):
		self.n = n
		self.k = k
		self.board = np.zeros((n**2,))
		self.qs = np.full((n**2, k), -1)

	def get_n_queens(self):
		return int(self.board.sum())

	def get_board(self):
		total_string = ""
		for row in self.board.reshape(self.n, self.n):
			row_string = ""
			for col in row:
				if col == 0:
					row_string += ". "
				elif col == 1:
					row_string += "Q "
			total_string += row_string + "\n"
		return total_string

	def	get_qs(self):
		total_string = ""
		for i, row in enumerate(self.qs):
			row_string = f"{i+1}:	"
			for col in row:
				if col == -1:
					row_string += "_ "
				elif col == 0:
					row_string += ". "
				elif col == 1:
					row_string += "1 "
			total_string += row_string + "\n"
		return total_string
	
	def __str__(self):
		return str(self.board.reshape((self.n, self.n)))
	
	def __repr__(self):
		return self.__str__()


def range_incl(start, stop):
	return range(start, stop+1)

def xy_to_i(x, y, n):
	return (x-1)*n+y	

class Get_Index:
	def __init__(self, n) -> None:
		self.n = n

	def __call__(self, x, y) -> int:
		return xy_to_i(x, y, self.n)


def define_psi(n):
	N = n**2

	b = {i: sympy.Symbol('b_' + str(i)) for i in range_incl(1, N)}
	idx = Get_Index(n)

	n_clauses = 0

	psi_n_clauses = []

	# Restrict rows
	n_clauses += n*n*(n-1)/2
	for i in range_incl(1, n):
		for j in range_incl(1, n):
			for l in range_incl(j+1, n):
				psi_n_clauses.extend([
					Not(b[idx(i,j)]) | Not(b[idx(i,l)]), # same row
				])
	# Restrict columns
	n_clauses += n*n*(n-1)/2
	for i in range_incl(1, n):
		for j in range_incl(1, n):
			for k in range_incl(i+1, n):
				psi_n_clauses.extend([
					Not(b[idx(i,j)]) | Not(b[idx(k,j)]), # same column
				])
	# Restrict \ diagonal
	n_clauses += n*n*(n-1)/2
	for i in range_incl(1, n):
		for j in range_incl(1, n):
			for m in range_incl(1, min(n-i, n-j)):
				psi_n_clauses.extend([
					Not(b[idx(i,j)]) | Not(b[idx(i+m,j+m)]),
				])
	# Restrict / diagonal
	n_clauses += n*n*(n-1)/2
	for i in range_incl(1, n):
		for j in range_incl(1, n):
			for m in range_incl(1, min(n-i, j)):
				psi_n_clauses.extend([
					Not(b[idx(i,j)]) | Not(b[idx(i+m,j-m)]),
				])				

	print(f"#clauses: {len(psi_n_clauses)} estimated: {n_clauses}")

	psi_n = And(*psi_n_clauses)
	return psi_n


def define_chi(n, k):
	N = n**2
	k = min(k, N)

	b = {i: sympy.Symbol('b_' + str(i)) for i in range_incl(1, N)}
	q = {i:{j: sympy.Symbol(f'q_{i}_{j}') 
			for j in range_incl(1, min(i,k))} 
		for i in range_incl(1, N)}

	n_clauses = 0

	n_clauses += 2
	chi_n_k_clauses = [
		q[N][k],				# Force k queens
		Not(q[1][1]) | b[1],	# Force q -> b for 1,1
	]

	# Force ( q[i][j] -> q[i-1][j] | b[i]&q[i-1][j-1] ) paths in q table
	# Every q between first column and diagonal
	n_clauses += 2 * ( (N-k)*(k-1) + (k-2)*(k-1)/2 ) # taking worst case
	for i in range_incl(2, N):
		for j in range_incl(2, min(i-1, k)):
			chi_n_k_clauses.extend([
				(Not(q[i][j]) | q[i-1][j] | q[i-1][j-1]),
				(Not(q[i][j]) | q[i-1][j] | b[i]),
			])
	# Diagonal qs except 1,1
	n_clauses += 2 * (k-1)
	for i in range_incl(2, k):
		chi_n_k_clauses.extend([
			(Not(q[i][i]) | q[i-1][i-1]),
			(Not(q[i][i]) | b[i]),
		])
	# First column of qs except 1,1
	n_clauses += (N-1)
	for i in range_incl(2, N):
		chi_n_k_clauses.append(
			(Not(q[i][1]) | q[i-1][1] | b[i])
		)

	# ----------------------------------------------------------------
	# Optional for functionality but nice for visualizing the q table.
	# Also speeds up computation by limiting combinations.
	# Force q monotonicity
	n_clauses += (N-1-k)*k + k*(k+1)/2
	for i in range_incl(1, N-1):
		for j in range_incl(1, min(i, k)):
			chi_n_k_clauses.append(
				(Not(q[i][j]) | q[i+1][j])
			)
	# ----------------------------------------------------------------

	# ----------------------------------------------------------------
	# Optional for functionality but speeds up computation by 
	# limiting combinations.
	# Force b -> q
	# b[i] & q[i-1][j-1] -> q[i][j]
	n_clauses += (N-k)*(k-1) + (k-1)*k/2
	for i in range_incl(2, N):
		for j in range_incl(2, min(i, k)):
			chi_n_k_clauses.append(
				(Not(b[i]) | Not(q[i-1][j-1]) | q[i][j])
			)
	# Force b -> q for first column
	n_clauses += (N-1)
	for i in range_incl(1, N-1):
		chi_n_k_clauses.append(
				Not(b[i]) | q[i][1]
		)
	# ----------------------------------------------------------------

	print(f"#clauses: {len(chi_n_k_clauses)} estimated: {n_clauses}")

	chi_n_k = And(*chi_n_k_clauses)
	return chi_n_k


if __name__ == "__main__":
	n = 8
	k = 8
	psi_n = define_psi(n)		# ~ 2 n^3 clauses
	chi_n_k = define_chi(n, k) 	# ~ 4kn^2 clauses

	formula = And(
		psi_n,
		chi_n_k
	)

	if n <= 4:
		solutions = list(sympy.logic.inference.satisfiable(formula, all_models=True))
	else:
		solutions = [sympy.logic.inference.satisfiable(formula, all_models=False)]

	queen_amounts = []
	for sol in solutions:
		if not sol:
			print("unsat")
			break

		cb = ChessBoard(n, k)
		for symbol, true in sol.items():
			symbol = str(symbol)
			if 'b' in symbol:
				i = int(symbol.split('_')[1]) - 1
				cb.board[i] = int(true)
			elif 'q' in symbol:
				i = int(symbol.split('_')[1]) - 1
				j = int(symbol.split('_')[2]) - 1
				cb.qs[i][j] = int(true)

		queen_amount = cb.get_n_queens()
		queen_amounts.append(queen_amount)
		print(f"{queen_amount} Queens")
		if n <= 100:
			print(f"Board: \n{cb.get_board()}")
		if n <= 5:
			print(f"qs: \n{cb.get_qs()}")
		print()
	if len(solutions) > 1:
		print(f"#solutions: {len(solutions)}")
		print(f"min queens: {min(queen_amounts)}")
