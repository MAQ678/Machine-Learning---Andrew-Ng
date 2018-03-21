function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1 = X;

z2 =([ones(m,1) X] * Theta1');
a2 = sigmoid(z2);


z3 = [ones(size(a2,1),1) a2] * Theta2';
h = sigmoid(z3);

%J = (-1/m) * ( (y' * log(h)) - ((1- y') * log(1-h)));

yn = zeros(m,num_labels);

for i=1:m
	yn(i,y(i)) = 1;
end

% J = (-1/m) * sum( (yn' * log(h)) + ((1- yn') * log(1-h)));


for i=1:m
	for k=1:num_labels
		J = J + (yn(i,k) * log(h(i,k)) + (1-yn(i,k)) * log(1 - h(i,k) ) );
	end
end

TmpTheta1 = Theta1.^2;
TmpTheta2 = Theta2.^2;

J = J * (-1/m) + ((lambda/(2*m)) * ( sum(sum(TmpTheta1,1)) - sum(sum(TmpTheta1(:,1),1))  + sum(sum(TmpTheta2,1)) - sum(sum(TmpTheta2(:,1),1)) ));


% -------------------------------------------------------------
% Theta1 = 25*401
% Theta2 = 10*26
% size(Theta1)
bigDel_1 = zeros(size(Theta1));
bigDel_2 = zeros(size(Theta2));
for t=1:m
	a_1 = X(t,:)';
	a_1 = [1 ; a_1];

	z_2 = Theta1 * a_1;
	a_2 = sigmoid(z_2);
	a_2 = [1 ; a_2];

	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);


	delt_3 = a_3 - yn(t,:)'; % important phase (yn), here beware of using only yn(t) or yn(t,:)
							 % if you use yn(t,:), then dimension is 1 * 10 , whether dimension of
							 % a_3 is 10 * 1, and matlab will give you a result


	delt_2 = (Theta2(:,2:end)' * delt_3);	% an error will occur if you use  only Theta2 and in next line
										 	% you use delt_2 = delt_2(2:end);

	delt_2 = delt_2 .* sigmoidGradient(z_2);

	% size(a_1)
	bigDel_1 = bigDel_1 + delt_2 * a_1';

	bigDel_2 = bigDel_2 + delt_3 * a_2';
end
Theta1_grad = (1/m) * bigDel_1;
Theta2_grad = (1/m) * bigDel_2;

% =========================================================================
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
