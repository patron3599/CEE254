function pred_pm2d5 = pm2d5_pred_model(train_data, test_data, problem_type)
    %% Team 2
    %% Jack Chen, Peter Lee, Chris Patron, Owen Zukowski
    %% 12/2/2022
    
    %% process input data
    % combination of all sensors
    training_data_total = train_data;
    % find base time to create time_num array
    date_time_base = min(training_data_total.time);
    % find time_num array for test data
    test_data.time_num = (datenum(test_data.time-date_time_base));

    % initialize table
    training_data = {};
    training_data_filtered = {};

    % define retiming interval
    % short-term
    if problem_type == 1
        dt = seconds(60);
    % long-term    
    elseif problem_type == 2
        dt = seconds(60*60);
    % interpolation
    elseif problem_type == 3
        dt = seconds(60*5);
    end
    % indices to segment data into each sensor
    sensor_no = 0;
    start_index = 1;
    end_index = 0;
    
    % segment data into each sensor
    for i = 1:1:height(training_data_total)-1
        % next sensor segment occurs
        if training_data_total.time(i) > training_data_total.time(i+1)
            sensor_no = sensor_no + 1;
            end_index = i;
            % get corresponding training data
            data = training_data_total(start_index:end_index,:);
            % create time_num array
            data.time_num = (datenum(data.time-date_time_base));
            data.sensor = sensor_no*ones(height(data),1);
            training_data{sensor_no} = data;
            % use PSD to find threshold frequency
            pm2d5_detrend = detrend(data.pm2d5);
            fs = 1/3;
            [pxx,f] = pwelch(pm2d5_detrend,600,300,600,fs);
            transform = 10*log10(pxx);
            threshold_index = max(find(transform>40));
            threshold_freq = f(threshold_index);
            % apply filter
            cut = threshold_freq;
            stpns = 0.95;
            pm2d5_lowpassed = lowpass(data.pm2d5,cut,fs,'Steepness',stpns);
            data_filtered = data;
            data_filtered.pm2d5 = pm2d5_lowpassed;
            % apply retime
            data_filtered = table2timetable(data_filtered);
            data_filtered_retime = retime(data_filtered,'regular','mean','TimeStep',dt);
            data_filtered = timetable2table(data_filtered_retime);
            training_data_filtered{sensor_no} = data_filtered;
            start_index = i+1;
        end
    end
    
    % create filtered total table
    training_data_filtered_total = {};
    
    for i = 1:1:length(training_data_filtered)
        training_data_filtered_total = [training_data_filtered_total; training_data_filtered{i}];
    end

    % define new training data for use
    training_data = training_data_filtered_total;
    % corresponding PM2.5
    y_train = training_data.pm2d5;
    
    %% Problem 1: Short-term
    if problem_type ==1
        test_lon = test_data.lon([(1:4)*3]);
        test_lat = test_data.lat([(1:4)*3]);
        
        train_lon = zeros(5,1);
        train_lat = zeros(5,1);
        
        sensor = [1:5];
        
        for j = 1:length(sensor)
            temp = training_data(training_data.sensor==j,:);
            training_data_static_AR{j} = temp(1:end-5,:);
            
            train_lon(j) = training_data_static_AR{j}.lon(1);
            train_lat(j) = training_data_static_AR{j}.lat(1);
    
            ARIMAX_pm2d5 = arima('Constant',NaN,'ARLags',1:14,'D',0,'MALags',1:8,'Distribution','Gaussian');
            validIndices = find(~any(isnan([training_data_static_AR{sensor(j)}.pm2d5,training_data_static_AR{sensor(j)}.hmd,training_data_static_AR{sensor(j)}.tmp]),2));
            preSampleNumber = ARIMAX_pm2d5.P;
            preSampleResponse = training_data_static_AR{sensor(j)}.pm2d5(validIndices(1:preSampleNumber));
            estimateResponse = training_data_static_AR{sensor(j)}.pm2d5(validIndices(preSampleNumber+1:end));
            ARIMAX_pm2d5 = estimate(ARIMAX_pm2d5,estimateResponse,'Y0',preSampleResponse,'X',[training_data_static_AR{sensor(j)}.hmd,training_data_static_AR{sensor(j)}.tmp],'Display','off');
    
            [static_sensor_predict(:,j),YMSE] = forecast(ARIMAX_pm2d5,3*60,'Y0',training_data_static_AR{sensor(j)}.pm2d5(validIndices),'XF',[training_data_static_AR{sensor(j)}.hmd(validIndices), training_data_static_AR{sensor(j)}.tmp(validIndices)]);
        end
        
        distance = zeros(5,1);
        weight = zeros(5,1);
        
        for i=1:4
            for k=1:5
                distance(k) = sqrt((train_lon(k)-test_lon(i)).^2+(train_lat(k)-test_lat(i)).^2);
            end
            
            for k=1:5
                weight(k) = 1/distance(k)/(sum(1./distance));
            end
            
            y_test_predict(:,i) = static_sensor_predict*weight;
        end
        
        pred_pm2d5 = reshape(y_test_predict([60 120 180],:),[],1);
    end

    %% Problem 2: Long-term
    if problem_type == 2
        % remove NaN rows
        training_data(any(ismissing(training_data), 2), :) = [];
        % segment data into quadrants
        med_lat = median(train_data.lat);
        med_lon = median(train_data.lon);
        
        idx = training_data.lat >= med_lat & training_data.lon >= med_lon;
        training_data_quad{1} = training_data(idx,:);
        idx = training_data.lat <= med_lat & training_data.lon >= med_lon;
        training_data_quad{2} = training_data(idx,:);
        idx = training_data.lat <= med_lat & training_data.lon <= med_lon;
        training_data_quad{3} = training_data(idx,:);
        idx = training_data.lat >= med_lat & training_data.lon <= med_lon;
        training_data_quad{4} = training_data(idx,:);
        
        % create set of predictors
        predictorNames = {'tmp','hmd','time_num'};
        x_test = test_data(:,predictorNames);

        % create gaussian process model for each quadrant
        for i = 1:1:4
            training_data_ind = training_data_quad{i};
            x_train = training_data_ind(:,predictorNames);
            y_train = training_data_ind.pm2d5;

            predictors = training_data_ind(:, predictorNames);
            response = training_data_ind.pm2d5;

            % perform gaussian process regression
            gprMdl{i} = fitrgp(predictors,response,'BasisFunction', 'constant',...,
                'FitMethod','sr','PredictMethod','bcd','KernelFunction', 'matern32','Standardize', true);

            % make training prediction
            y_train_predict = predict(gprMdl{i},x_train); 
        end

        % make predictions
        for i = 1:1:height(test_data)
            if test_data.lat(i)>med_lat && test_data.lon(i)>med_lon
                y_test_predict(i,1) = predict(gprMdl{1},x_test(i,:));
            elseif test_data.lat(i)<med_lat && test_data.lon(i)>med_lon
                y_test_predict(i,1) = predict(gprMdl{2},x_test(i,:));
            elseif test_data.lat(i)<med_lat && test_data.lon(i)<med_lon
                y_test_predict(i,1) = predict(gprMdl{3},x_test(i,:));
            elseif test_data.lat(i)>med_lat && test_data.lon(i)<med_lon
                y_test_predict(i,1) = predict(gprMdl{4},x_test(i,:));
            end
        end
        pred_pm2d5 = y_test_predict;
    end
       
    %% Problem 3: Interpolation
    if problem_type == 3
        % filter out NaN rows
        training_data(any(ismissing(training_data), 2), :) = [];
        % segment data into quadrants
        med_lat = median(train_data.lat);
        med_lon = median(train_data.lon);
        
        idx = training_data.lat >= med_lat & training_data.lon >= med_lon;
        training_data_quad{1} = training_data(idx,:);
        idx = training_data.lat <= med_lat & training_data.lon >= med_lon;
        training_data_quad{2} = training_data(idx,:);
        idx = training_data.lat <= med_lat & training_data.lon <= med_lon;
        training_data_quad{3} = training_data(idx,:);
        idx = training_data.lat >= med_lat & training_data.lon <= med_lon;
        training_data_quad{4} = training_data(idx,:);
        
        % define predictor set
        predictorNames = {'tmp','hmd','time_num'};
        x_test = test_data(:,predictorNames);    

        % create gaussian process model for each quadrant
        for i = 1:1:4
            training_data_ind = training_data_quad{i};
            x_train = training_data_ind(:,predictorNames);
            y_train = training_data_ind.pm2d5;

            predictors = training_data_ind(:, predictorNames);
            response = training_data_ind.pm2d5;
            
            % perform gaussian process regression
            sigmaF0 = std(response);
            d = size(x_train,2);
            sigmaM0 = 10*ones(d,1);

            gprMdl{i} = fitrgp(predictors,response,'BasisFunction', 'linear','sigma',0.5,...,
                'PredictMethod','exact','KernelFunction', 'ardmatern32',...
                'KernelParameters',[sigmaM0;sigmaF0],'Standardize', true);

            % make training prediction    
            y_train_predict = predict(gprMdl{i},x_train); 
        end

        % make predictions
        for i = 1:1:height(test_data)
            if test_data.lat(i)>med_lat && test_data.lon(i)>med_lon
                test_data_quad_1 = 1;
                y_test_predict(i,1) = predict(gprMdl{1},x_test(i,:));
            elseif test_data.lat(i)<med_lat && test_data.lon(i)>med_lon
                y_test_predict(i,1) = predict(gprMdl{2},x_test(i,:));
            elseif test_data.lat(i)<med_lat && test_data.lon(i)<med_lon
                y_test_predict(i,1) = predict(gprMdl{3},x_test(i,:));
            elseif test_data.lat(i)>med_lat && test_data.lon(i)<med_lon
                y_test_predict(i,1) = predict(gprMdl{4},x_test(i,:));
            end
        end
        pred_pm2d5 = y_test_predict;
    end

end