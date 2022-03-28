classdef mytree
    methods(Static) 
            %每个图形A根据HOG函数进行特征提取之后是一个一维向量，假设特征数为n
            %有m个图形时，train_example就是m*n的矩阵,
            %train_labels是一个m*1的列向量，train_labels(1)=2表示第一个图属于第二类
            %假设每个图形为r*c的矩阵，则有n = r*c
        function m = fit(train_examples, train_labels ,r,c)
            
			emptyNode.number = [];
            emptyNode.examples = [];
            emptyNode.labels = [];
            emptyNode.prediction = [];
            emptyNode.impurityMeasure = [];
            emptyNode.children = {};
            emptyNode.splitFeature = [];
            emptyNode.splitFeatureName = [];
            emptyNode.splitValue = [];
            %存储节点的L(r*1)和R(c*1)向量
            emptyNode.L = [];
            emptyNode.R = [];
            
            %记录reshape的矩阵的行列
            emptyNode.r = r;
            emptyNode.c = c;
            
            emptyNode.Featurenum = [];
            m.emptyNode = emptyNode;
            
            r = emptyNode;
            r.number = 1;
            r.labels = train_labels;
            r.examples = train_examples;
            r.prediction = mode(r.labels);
            r.Featurenumber = size(train_examples,2);
            
            m.min_parent_size = 2;
            m.unique_classes = unique(r.labels);
            m.feature_names = train_examples.Properties.VariableNames;
			m.nodes = 1;
            m.N = size(train_examples,1);
            m.Featurenumber = size(train_examples,2);
            m.tree = mytree.trySplit(m, r);

        end
        
        function node = trySplit(m, node)
            %如果节点样本数少于阈值则直接返回
            if size(node.examples, 1) < m.min_parent_size
				return
            end
            %如果节点只有一类
            cn = unique(node.labels);%different classes
            l=length(cn);%totoal numbers of classes
            if l==1%if only one class,just use the class to be the lable of the tree and return
                return 
            end
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);
            %对节点的数据进行处理，先调用iterative2DLDA函数找到该节点最佳的L和R，再对每个数据进行投影
            %在iterative2DLDA函数中，训练数据集是一个矩阵，每一列表示一个数据。标签是一个列向量，每i行表示第i个数据属于哪一列
            trains = table2array(node.examples);
            trainset = trains';
            %将标签数字化,得到一个列向量标签
            label = numlabels(node.labels);
            %调用iterative2DLDA
            [node.R,node.L] = iterative2DLDA(trainset, label, 1,1,node.r,node.c);
            pi = [];
                 for i=1:size(trains,1)
                     pi1 = reshape(trains(i,:),[node.r,node.c]);
                     %将节点的每个数据进行投影，投影成一维数据
                     pi(i) = node.L'*pi1*node.R;
                 end
                 %将投影后的数升序排列，原数据examples和label也按照索引重新排列得到es ls
                [ps,n] = sort(pi);
                es = node.examples(n,:);
                ls = node.labels(n);
                biggest_reduction = -Inf;
                biggest_reduction_index = -1;
                biggest_reduction_value = NaN;
                %根据每个pi进行划分，找出能使不纯度降低最大的pi作为该节点的划分投影属性
                for j=1:(size(ps,2)-1)
                    if ps(j) == ps(j+1)
                        continue;
                    end
                    %fprintf('evaluating possible splits on artificial feature %d/%d\n', i, size(node.examples,1));
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));
                    if this_reduction > biggest_reduction
                        biggest_reduction = this_reduction;
                        biggest_reduction_index = j;
                    end
                end
              %找到最佳的pi的索引
            winning_reduction = biggest_reduction;
            winning_index = biggest_reduction_index;
            if winning_reduction <= 0
                return
            else
                %es为排序后的examples ls为排序后的标签
                node.splitValue = ps(winning_index) + ps(winning_index+1) / 2;
                %划分节点时，小于pi的为1节点，大于的为2节点
                node.examples = [];
                node.labels = []; 
                node.prediction = [];

                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{1}.number = m.nodes;
                node.children{1}.examples = es(1:winning_index,:);
                node.children{1}.labels = ls(1:winning_index);
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{2}.number = m.nodes;
                node.children{2}.examples = es((winning_index+1):end,:);
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end
        
        function e = weightedImpurity(m, labels)
            weight = length(labels) / m.N;
            summ = 0;
            obsInThisNode = length(labels);
            for i=1:length(m.unique_classes)
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pc*pc);
			end
            g = 1 - summ;
            e = weight * g;

        end

        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            for i=1:size(test_examples,1)
                
				%fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                this_test_example = test_examples(i,:);
                this_prediction = mytree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
			end
        end

        function prediction = predict_one(m, this_test_example)
            
			node = mytree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
        
		end
        
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children)
                return;
            else
                test1 = table2array(this_test_example);
                testtemp = reshape(test1,[node.r,node.c]);
                test = node.L'*testtemp*node.R;
                if test < node.splitValue
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});        
            end
        
		end
		
    end
end
  function [label] = numlabels(labels)
                la = unique(labels);
                for i = 1:size(labels)
                    for j = 1:size(la)
                        if labels(i) == la(j)
                            label(i) = j;
                        end
                    end
                end
                label = label';
  end