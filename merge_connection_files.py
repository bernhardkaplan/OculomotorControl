import utils
import sys
import CreateConnections

params = utils.load_params(sys.argv[1])
CC = CreateConnections.CreateConnections(params, dummy=True)
CC.merge_connection_files(params)
