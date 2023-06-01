classdef CustomPolynomialNonLinearLayer < nnet.layer.Layer
    properties
        %
        % polynomial fitted data
        P0

        %
        % derivative of polynomial fitted data
        D0

        %
        % Needed properties
        ss
    end

    %
    % Unused properties, the Custom Polynomial Layer will not
    % learn
    properties (Learnable)
    end
    properties (State)
    end
    properties (Learnable, State)
    end

    methods
        function layer = CustomPolynomialNonLinearLayer (Name, poly, der, ss)
            layer.Name       = Name;
            layer.NumInputs  = 1;
            layer.NumOutputs = 1;
            layer.ss         = ss;

            layer.P0         = poly;
            layer.D0         = der;
            
        end

        function Z = predict(layer, X)
            Z = zeros(size(X),'like',X);
            for i=1:size(X,4)
                Z(:,:,1,i) = polyvalm(layer.P0, X(:,:,1,i));
            end
        end

        function dLdX = backward(layer, X, Z, dLdZ, dLdSout)
            dZdX = zeros(size(X),'like',X);
            for i=1:size(X,4)
                dZdX(:,:,1,i) = polyvalm(layer.D0, X(:,:,1,i));
            end
            dLdX = dLdZ .* dZdX;
        end
    end
end