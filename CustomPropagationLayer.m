classdef CustomPropagationLayer < nnet.layer.Layer %  & nnet.layer.Acceleratable 
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        w
        wc

        Nx
        Ny
        nx
        ny
        d
        wv
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = CustomPropagationLayer(Name, Nx, Ny, nx, ny, d, wv)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = 1;
            layer.NumOutputs = 1;

            layer.Nx = Nx;
            layer.Ny = Ny;
            layer.nx = nx;
            layer.ny = ny;
            layer.d  = d;
            layer.wv = wv;
            layer = layer.compute_w();
        end

        function layer = compute_w(layer)
            dx = layer.nx/layer.Nx;
            dy = layer.ny/layer.Ny;
            
            rangex = 1/dx;  % number of frequencies available
            rangey = 1/dy;

            posx = linspace(-rangex/2, rangex/2, layer.Nx);
            posy = linspace(-rangey/2, rangey/2, layer.Ny);

            [fxx, fyy] = meshgrid(posy, posx);
    
            kz = 2 * pi * sqrt((1/layer.wv)^2 -(fxx.^2)-(fyy.^2));

            layer.w  = exp(1i * kz * layer.d);
            layer.wc = layer.w';
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            % normalize and apply layer weights
            R = gpuArray(cast(X, 'like', 1));
            Z = gpuArray(zeros(size(R), 'double'));
            for i=1:size(R, 4)
                Z(:,:,1,i)=abs(ifft2(fft2(R(:,:,1,i)) .* layer.w));
            end
            Z = cast(Z, 'like', X);
        end

        % function dLdX = backward(layer, X, Z, dLdZ, dLdSout)
        %     % (Optional) Backward propagate the derivative of the loss
        %     % function through the layer.
        %     %
        %     % Inputs:
        %     %         layer   - Layer to backward propagate through 
        %     %         X       - Layer input data 
        %     %         Z       - Layer output data 
        %     %         dLdZ    - Derivative of loss with respect to layer 
        %     %                   output
        %     %         dLdSout - (Optional) Derivative of loss with respect 
        %     %                   to state output
        %     %         memory  - Memory value from forward function
        %     % Outputs:
        %     %         dLdX   - Derivative of loss with respect to layer input
        %     %         dLdW   - (Optional) Derivative of loss with respect to
        %     %                  learnable parameter 
        %     %         dLdSin - (Optional) Derivative of loss with respect to 
        %     %                  state input
        %     %
        %     %  - For layers with state parameters, the backward syntax must
        %     %    include both dLdSout and dLdSin, or neither.
        %     %  - For layers with multiple inputs, replace X and dLdX with
        %     %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
        %     %    the number of inputs.
        %     %  - For layers with multiple outputs, replace Z and dlZ with
        %     %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
        %     %    number of outputs.
        %     %  - For layers with multiple learnable parameters, replace 
        %     %    dLdW with dLdW1,...,dLdWP, where P is the number of 
        %     %    learnable parameters.
        %     %  - For layers with multiple state parameters, replace dLdSin
        %     %    and dLdSout with dLdSin1,...,dLdSinK and 
        %     %    dLdSout1,...,dldSoutK, respectively, where K is the number
        %     %    of state parameters.
        % 
        %     % Define layer backward function here.
        %     dLdX = gpuArray(zeros(size(X), 'double'));
        %     R    = gpuArray(cast(X, 'like', 1));
        % 
        % 
        %     for i=1:size(X, 4)
        %         A = ifft2(fft2(R(:,:,1,i)) .* real(layer.w));
        %         B = ifft2(fft2(R(:,:,1,i)) .* imag(layer.w));
        %         C = sqrt(A.^2+B.^2);
        %         dZdA = 2 * A ./ C;
        %         dZdB = 2 * B ./ C;
        % 
        %         % real
        %         dAdX = ifft2(fft2(dLdZ) .* real(layer.w));
        % 
        %         % imag
        %         dBdX = ifft2(fft2(dLdZ) .* -imag(layer.w));
        % 
        %         dLdX(:,:,1,i)=dLdZ .* ((dZdA .* dAdX) + (dZdB .* dBdX));
        %     end
        % 
        %     dLdX = cast(dLdX, 'like', dLdZ);
        % end
    end
end