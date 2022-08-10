classdef EvoXBaseProblem < PROBLEM
    properties (Access = private)
        conn
        options
        true_eval
    end
    properties
        test
    end
    methods
        function delete(obj)
            if ~isempty(obj.conn)
                request.operation = 'delete';
                try
                    obj.conn.writeline(jsonencode(request));
                    obj.conn.flush('output');
                catch
                    % do nothing here
                end
            end
        end
        function InitRemoteObj(obj, config)
            request.config = config;
            request.operation = 'create';
            obj.conn = tcpclient('127.0.0.1', 9876, 'timeout', 30);
            obj.conn.writeline(jsonencode(request));
            obj.conn.flush('output');
            response = jsondecode(obj.conn.readline());
            if ~strcmp(response.status, 'ok')
                exception = MException('EvoXBase:createFailed','Failed to connect to remote server');
                throw(exception);
            end
        end

        function Setting(obj, config)
            obj.InitRemoteObj(config);
            query.operation = 'settings';
            obj.conn.writeline(jsonencode(query));
            obj.conn.flush('output');
            response = jsondecode(obj.conn.readline());
            if strcmp(response.status, 'ok')
                basicinfo = response.result;
            else
                exception = MException('TestFunction:queryFailed','Failed to query');
                throw(exception);
            end
            obj.M = basicinfo.n_obj;
            if isempty(obj.D); obj.D = basicinfo.n_var; end
            obj.lower    = reshape(basicinfo.lb, 1, []);
            obj.upper    = reshape(basicinfo.ub, 1, []);
            obj.encoding = 'real';
            obj.true_eval = false;
        end
        
        function r = CalObj(obj, x)
            query.operation = 'query';
            query.encoding = x;
            query.true_eval = obj.true_eval;
            obj.conn.writeline(jsonencode(query));
            obj.conn.flush('output');
            response = jsondecode(obj.conn.readline());
            if strcmp(response.status, 'ok')
                r = response.result;
            else
                exception = MException('EvoXBench:queryFailed','Failed to query');
                throw(exception);
            end
        end

        function PopDec = CalDec(obj, PopDec)
            % clip and round, because evoxbench uses discrete encoding
            PopDec = min(PopDec, repmat(obj.upper, size(PopDec,1), 1));
            PopDec = max(PopDec, repmat(obj.lower, size(PopDec,1), 1));
            PopDec = round(PopDec);

            % deduplicate
            TotalLen = size(PopDec, 1);
            [~, ia] = unique(PopDec, 'rows');
            NonDupLen = size(ia, 1);
            DupLen = TotalLen - NonDupLen;
            DupIndex = true(TotalLen, 1);
            DupIndex(ia) = false;
            PopDec(DupIndex, :) = obj.Sample(DupLen);
        end

        function r = Sample(obj, n_samples)
            query.operation = 'sample';
            query.n_samples = n_samples;
            obj.conn.writeline(jsonencode(query));
            obj.conn.flush('output');
            response = jsondecode(obj.conn.readline());
            if strcmp(response.status, 'ok')
                r = response.result;
            else
                exception = MException('EvoXBench:queryFailed','Failed to query');
                throw(exception);
            end
        end

        function r = GetMetric(obj, inputs, indicator)
            % indicator is one of 'igd', 'hv', 'normalized_hv'
            F = obj.CalObj(inputs);
            invalidSolutions = any(F == inf, 2);
            validSolutions = ~invalidSolutions;
            inputs = inputs(validSolutions, :);
            query.operation = 'calc_perf_indicator';
            query.inputs = inputs;
            query.indicator = indicator;
            obj.conn.writeline(jsonencode(query));
            obj.conn.flush('output');
            response = jsondecode(obj.conn.readline());
            if strcmp(response.status, 'ok')
                r = response.result;
            else
                exception = MException('EvoXBench:queryFailed','Failed to query');
                throw(exception);
            end
        end

        function r = GetOptimum(obj, N)
            query.operation = 'pareto_front';
            query.n_samples = N;
            obj.conn.writeline(jsonencode(query));
            obj.conn.flush('output');
            response = jsondecode(obj.conn.readline());
            if strcmp(response.status, 'ok')
                r = response.result;
            else
                exception = MException('EvoXBench:queryFailed','Failed to query');
                throw(exception);
            end
        end

        function r = GetPF(obj)
            %% -1 will get all points
            r = obj.GetOptimum(-1);
        end

        function SetValid(obj)
            % set to validation phase
            obj.true_eval = true;
        end

        function SetTest(obj)
            % set to test time phase
            obj.true_eval = true;
        end
    end
end
