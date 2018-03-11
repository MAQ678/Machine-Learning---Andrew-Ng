function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


z2 = ([ones(m, 1) X] * Theta1');
a2 = sigmoid(z2);
% for i=1:m
% 	[tmp,a2(i)] = max(z(i,:),[],2);
% end

% size(a2)


z3 = ([ones(size(a2,1), 1) a2] * Theta2');

a3 = sigmoid(z3);
size(a3)
pause;

for i=1:m
	[tmp,p(i)] = max(a3(i,:),[],2);
end



% =========================================================================


end
