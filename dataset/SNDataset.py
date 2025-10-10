from torch_geometric.data import Dataset,Data
import numpy as np
import torch
import os
from copy import deepcopy
import random
from tqdm import tqdm
from Microservices.AnomalyDetection.DALAD.ADGS.ADGS \
    import Event_masking,Event_Metric_masking,invocation_Metric_masking,invocation_swap_nodes,invocation_interruption
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SNDataset(Dataset):
    def __init__(self,root,transform=None,pre_transform=None):
        super(SNDataset, self).__init__(root,transform,pre_transform)

    @property
    def raw_file_names(self):
        graphfile="trace_graph_depth10.txt"
        logfile="log_template_depth10.txt"
        logvariablefile="log_variable_template_depth10_2processed.txt"

        vectortemplatefile="log_template_vector300_depth10.txt"
        logvariableVectorTemplateFile="log_variable_vector300_depth10_2processed.txt"
        SpanTemplateIdfile="SpanTemplateId_depth10.txt"

        relation_servicefile='relation_service_kpi.txt'
        servicefile='service_kpi.txt'

        return [graphfile,logfile,logvariablefile,vectortemplatefile,
                logvariableVectorTemplateFile,SpanTemplateIdfile,relation_servicefile,servicefile]

    @property
    def processed_file_names(self):
        return ['data_0_depth10.pt']

    def download(self):
        pass

    def process(self):
        All_trace = set()
        with open("./OBD/SN/raw/trace_OCV.txt", "r", encoding='utf-8') as f:
            lines = f.readlines()
            for l in tqdm(lines):
                l = l.replace("\n", "").strip().split(":")
                traceid = l[0]
                All_trace.add(traceid)
        f.close()

        edge_type_dict = {'Sequence': 0, 'Same_Process_Synchronous_Request': 1, 'Same_Process_Synchronous_Response': 2,
                          'Cross_Process_Synchronous_Request': 3, 'Cross_Process_Synchronous_Response': 4,
                          'Asynchronous_Request': 5}

        service_metric={}
        service_relation_metric={}
        with open(self.raw_paths[7], "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "").strip().split()
                service_name = line[0]
                timestamp = line[1]
                features = line[2:]
                service_metric[(service_name, timestamp)] = list(map(float, features))
        f.close()

        with open(self.raw_paths[6], "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "").strip().split()
                service_name1 = line[0]
                service_name2 = line[1]
                timestamp = line[2]
                features = line[3:]
                service_relation_metric[(service_name1, service_name2, timestamp)] = list(map(float, features))
                service_relation_metric[(service_name1, service_name1, timestamp)] = [-1] * 17
                service_relation_metric[(service_name2, service_name2, timestamp)] = [-1] * 17
        f.close()

        template_vector_dict = {}
        variable_vector_dict ={}
        SpanTemplateId=set()
        with open(self.raw_paths[5], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip()
                SpanTemplateId.add(line)
        f.close()

        with open(self.raw_paths[4], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip().split()
                id = line[0]
                variabelVector = line[1:]
                variable_vector_dict[id] = list(map(float,variabelVector))
        f.close()

        with open(self.raw_paths[3], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip().split()
                id = line[0]
                templateVector = line[1:]
                template_vector_dict[id] = list(map(float,templateVector))
        f.close()

        logVector={}
        logType={}
        logServiceName={}
        logTimeStamp = {}
        logSpanId={}

        index = 0
        with open(self.raw_paths[1], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().replace("\n", "").split(" - ")
                content = line[0].split("[SW_CTX:[")
                timestamp = content[0].replace("-", "").replace(" ", "").replace(":", "").replace(".", "")[:12]
                service_name = content[1].split(",")[0].strip()
                spanid = content[1].split(",")[3].split("]")[0].strip()
                template_id = line[1]

                logId = str(index)
                logVector[logId] = template_vector_dict[template_id]
                logTimeStamp[logId] = timestamp
                logServiceName[logId] = service_name
                logSpanId[logId] = spanid

                if template_id in SpanTemplateId:
                    logType[logId] = "SpanEvent"
                else:
                    logType[logId] = "LogEvent"
                index += 1
        f.close()

        variabelVector={}
        with open(self.raw_paths[2], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().replace("\n", "").split()
                variabeltemplate_id = line[1]
                logId = line[0]
                variabelVector[logId] = variable_vector_dict[variabeltemplate_id]
        f.close()

        idx=0
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                SpanEventNode = set()
                LogEventNode = set()

                line = line.strip().replace("\n", "").replace("\'", "").split("           ")
                trace_id = line[0]
                if trace_id not in All_trace:
                    continue

                edges = line[1][2:-2].split("), (")
                classlabel = int(line[2])
                isAnomaly = int(line[3])

                for edge in edges:
                    edge = edge.split(", ")
                    node1 = edge[0]
                    node2 = edge[1]
                    if logType[node1] == 'SpanEvent':
                        SpanEventNode.add(node1)
                    else:
                        LogEventNode.add(node1)

                    if logType[node2] == 'SpanEvent':
                        SpanEventNode.add(node2)
                    else:
                        LogEventNode.add(node2)

                EventX = []
                VariableX = []
                ServiceX = []
                NodeType = []
                for logId in LogEventNode:
                    EventX.append(logVector[logId])
                    VariableX.append(variabelVector[logId])
                    NodeType.append(0)

                    if (logServiceName[logId], logTimeStamp[logId]) in service_metric:
                        serviceX = service_metric[(logServiceName[logId], logTimeStamp[logId])]
                    else:
                        serviceX = [-1] * 9
                    ServiceX.append(serviceX)

                for logId in SpanEventNode:
                    EventX.append(logVector[logId])
                    VariableX.append(variabelVector[logId])
                    NodeType.append(1)

                    if (logServiceName[logId], logTimeStamp[logId]) in service_metric:
                        serviceX = service_metric[(logServiceName[logId], logTimeStamp[logId])]
                    else:
                        serviceX = [-1] * 9
                    ServiceX.append(serviceX)

                LogEventNode = list(LogEventNode)
                SpanEventNode = list(SpanEventNode)
                AllNode = LogEventNode + SpanEventNode

                Pre_edge_index = []
                Tar_edge_index = []

                EdgeType = []
                EdgeAttr = []

                for edge in edges:
                    edge = edge.split(", ")
                    node1 = edge[0]
                    node2 = edge[1]

                    edge_type = edge_type_dict[str(edge[2])]
                    EdgeType.append(edge_type)

                    node1_Id = AllNode.index(node1)
                    node2_Id = AllNode.index(node2)
                    Pre_edge_index.append(node1_Id)
                    Tar_edge_index.append(node2_Id)

                    service_name1 = logServiceName[node1]
                    service_name2 = logServiceName[node2]
                    if (service_name1, service_name2, logTimeStamp[node1]) in service_relation_metric:
                        target = service_relation_metric[(service_name1, service_name2, logTimeStamp[node1])]
                    elif (service_name2, service_name1, logTimeStamp[node1]) in service_relation_metric:
                        target = service_relation_metric[(service_name2, service_name1, logTimeStamp[node1])]
                    else:
                        target = [-1] * 17

                    EdgeAttr.append(target)

                labels = []
                labels.append([isAnomaly, classlabel])

                edge_index = torch.as_tensor([Pre_edge_index, Tar_edge_index], dtype=torch.long)
                X = torch.as_tensor(EventX, dtype=torch.float64)
                variableX = torch.as_tensor(VariableX, dtype=torch.float64)
                SerX = torch.as_tensor(ServiceX, dtype=torch.float64)
                NodeType = torch.as_tensor(NodeType, dtype=torch.long)
                EdgeType = torch.as_tensor(EdgeType, dtype=torch.long)
                EdgeAttr = torch.as_tensor(EdgeAttr, dtype=torch.float64)
                y = torch.as_tensor(labels, dtype=torch.long)

                data = Data(x=X, variableX=variableX, ServiceX=SerX, node_type=NodeType,
                            edge_type=EdgeType, edge_index=edge_index, edge_attr=EdgeAttr, y=y)

                torch.save(data, os.path.join(self.processed_dir, 'data_{}_depth10.pt'.format(idx)))
                idx += 1
        f.close()

    def len(self) -> int:
        datalen=0
        basedir="./OBD/SN/processed"
        for file in os.listdir(basedir):
            datalen+=1
        return datalen-2

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}_depth10.pt'))
        random_choice=random.randint(0,4)
        if random_choice==0:
            data_pos = Event_masking(deepcopy(data))
        elif random_choice==1:
            data_pos=invocation_interruption(deepcopy(data))
        elif random_choice==2:
            data_pos= invocation_swap_nodes(deepcopy(data))
        elif random_choice==3:
            data_pos = invocation_Metric_masking(deepcopy(data))
        else:
            data_pos = Event_Metric_masking(deepcopy(data))
        return data,data_pos






