
////////////////////////////////////////////////////////////////////////////////////////////////////
// replace each element of an array arr[i_elem] with its modulo with modulus, arr[i_elem] % modulus
////////////////////////////////////////////////////////////////////////////////////////////////////
template < class T >
__global__ void
moduloKernel
(T* array, int64_t n_elem, int64_t modulus)
{
  int64_t i_elem = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_elem >= n_elem ) {
    return;
  }
  array[i_elem] = array[i_elem] % modulus;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel that subtracts an offset and convert the source node indexes
// to the ConnKeyT representation in a range of connections in a block
template < class T, class ConnKeyT >
__global__ void
subtractSourceOffsetKernel
( ConnKeyT* conn_key_subarray, int64_t n_conn, int64_t offset, T source)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn ) {
    return;
  }
  int64_t index = (int64_t)conn_key_subarray[ i_conn ] - offset;
  inode_t i_source = getNodeIndex( source, index );
  setConnSource< ConnKeyT >( conn_key_subarray[ i_conn ], i_source );
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build connections with fixed indegree rule for source neurons and target neurons distributed across
// MPI processes (hosts)
// Template
///////////////////////////////////////////////////////////////////////////////////////////////////////
template < class ConnKeyT, class ConnStructT >
template < class T1, class T2 >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::_ConnectDistributedFixedIndegree
(int *source_host_arr, int n_source_host, T1* h_source_arr, int *n_source_arr,
 int *target_host_arr, int n_target_host, T2 *h_target_arr, int *n_target_arr,
 int indegree, int i_host_group, SynSpec &syn_spec)
{
  // check that connection key part has the right size in bytes
  if (sizeof(ConnKeyT) != 4 &&  sizeof(ConnKeyT) != 8) {
        throw ngpu_exception( "sizeof(ConnKeyT) must be either 4 or 8 bytes"
			      " in  _ConnectDistrubutedFixedIndegree" );
  }
  if (first_connection_flag_ == true) {
    remoteConnectionMapInit();
    first_connection_flag_ = false;
  }

  if (i_host_group<0) { // point-to-point MPI communication
    throw ngpu_exception( "_ConnectDistrubutedFixedIndegree is not available for point-to-point MPI communication" );
  }
  group_local_id = host_group_local_id_[i_host_group];
  if (group_local_id < 0) { // this host is not in group
    return 0;
  }

  // Connections must be created only for target_host == this_host
  // so first of all check if this_host is in target_host_arr, and in this case
  // find the index in the array and check that it is not present more than once
  int i_target_host = -1;
  for (int ith=0; ith<n_target_host; ith++) {
    int target_host = target_host_arr[ith];
    if ( target_host >= n_hosts_ ) {
      throw ngpu_exception( "Target host index out of range in _ConnectDistributedFixedIndegree" );
    }
    if ( target_host == this_host_ ) {
      if (i_target_host >= 0) {
	throw ngpu_exception( "Same target host repeated more than once in _ConnectDistributedFixedIndegree" );
      }
      i_target_host = ith;
    }
  }

  // arrays of source and target nodes of each source and target host in GPU memory
  // They can be either the index of the first node of each source/target group
  // or pointers to arrays of indexes
  T1 d_source_arr[n_source_host];
  T2 d_target_arr[n_target_host];

  // loop on source hosts
  for (int ish=0; ish<n_source_host; ish++) {
    // find the position i_host of the source host in the host group
    int source_host = source_host_arr[ish];
    if ( source_host >= n_hosts_ ) {
      throw ngpu_exception( "Source host index out of range in _ConnectDistributedFixedIndegree" );
    }
    // find the source host index in the host group
    auto it = std::find(host_group_[group_local_id].begin(), host_group_[group_local_id].end(), source_host);
    if (it == host_group_[group_local_id].end()) {
      throw ngpu_exception( "source host not found in host group in _ConnectDistrubutedFixedIndegree" );
    }
    int i_host = it - host_group_[group_local_id].begin();

    // insert the source node indexes in the host-group array of source nodes of the host i_host  
    for (inode_t i=0; i<n_source_arr[ish]; i++) {
      inode_t i_source = hGetNodeIndex(h_source_arr[ish], i);
      host_group_source_node_[group_local_id][i_host].insert(i_source);
    }
  }

  if (i_target_host < 0) {
    return; // this_host_ is not among target hosts
  }

  for (int ish=0; ish<n_source_host; ish++) {
    // if T1 is a pointer copy source index array from CPU to GPU memory
    // otherwise d_source_arr[ish] will be the first index of the source node group
    d_source_arr[ish] = copyNodeArrayToDevice( h_source_arr[ish], n_source_arr[ish] );
  }
  int n_target = n_target_arr[i_target_host];
  // if T2 is a pointer copy target index array from CPU to GPU memory
  // otherwise d_target will be the first index of the target node group
  inode_t* d_target = copyNodeArrayToDevice( h_target_arr[i_target_host], n_target );
  

  // compute number of new connections that must be created
  int64_t n_new_conn_tot = n_target*indegree;

  // Create new connection blocks as needed
  int new_n_block = ( int ) ( ( n_conn_ + n_new_conn_tot + conn_block_size_ - 1 ) / conn_block_size_ );
  allocateNewBlocks( new_n_block );

  // Cumulative sum of n_source: n_source_cumul
  int n_source_cumul[n_source_host + 1];
  n_source_cumul[0] = 0;
  for (int ish=0; ish<n_source_host; ish++) {
    n_source_cumul[ish+1] = n_source_cumul[ish] + n_source_arr[ish];
  }
  
  // total number of source nodes:
  n_source_tot = n_source_cumul[n_source_host];
  if (n_source_tot > (ConnKeyT)(-1)) { // for an unsigned integer type, -1 is the max value it can represent
    throw ngpu_exception( "Total number of source neuron too large for the connection key type"
			  " used in _ConnectDistrubutedFixedIndegree" );
  }
  
  // printf("Generating connections with fixed_indegree rule...\n");

  // Loop on connection blocks where new connections are stored
  //int64_t conn_source_ids_offset = 0; uncomment only if needed
  int64_t n_prev_conn = 0;
  int ib0 = ( int ) ( n_conn_ / conn_block_size_ );
  for ( int ib = ib0; ib < new_n_block; ib++ )
  {
    int64_t n_block_conn; // number of new connections in the current block of the loop
    int64_t i_conn0;      // index of first new connection in this block
    if ( new_n_block == ib0 + 1 ) // no new blocks were needed/allocated
    { // all connections are in the same block
      i_conn0 = n_conn_ % conn_block_size_;
      n_block_conn = n_new_conn_tot;
    }
    else if ( ib == ib0 ) // first block of the loop, cannot be the last (see above)
    { // first block
      i_conn0 = n_conn_ % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if ( ib == new_n_block - 1 ) // last block of the loop, cannot be the first (see above)
    { // last block
      i_conn0 = 0;
      n_block_conn = ( n_conn_ + n_new_conn_tot - 1 ) % conn_block_size_ + 1;
    }
    else // block is neither the first nor the last of the loop
    {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    // Connection source host&node indexes are extracted as 32 bit or 64 bit random integers
    // and then replaced with their modulo in the range from 0 to n_source_tot - 1 .
    // Each target node index is repeated indegree times in the connections.
    if (sizeof(ConnKeyT)==8) { // 64 bit
      // generate array of 64 bit random unsigned integers
      curandGenerateLongLong(conn_random_generator_[ this_host_ ][ this_host_ ],
			     (unsigned long long*)conn_key_vect_[ ib ] + i_conn0,
			     n_block_conn*sizeof(unsigned long long));
      // Replace each 64 bit random integer with its modulo in the range from 0 to n_source_tot - 1
      moduloKernel< unsigned long long > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>
	((unsigned long long*)conn_key_vect_[ ib ] + i_conn0, n_block_conn, n_source_tot);
      DBGCUDASYNC;
    }
    else {
      // generate array of 32 bit random unsigned integers
      curandGenerate(conn_random_generator_[ this_host_ ][ this_host_ ],
		     (unsigned int*)conn_key_vect_[ ib ] + i_conn0, n_block_conn*sizeof(unsigned int));
      // Replace each 32 bit random integer with its modulo in the range from 0 to n_source_tot - 1
      moduloKernel< unsigned int > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>
	((unsigned int*)conn_key_vect_[ ib ] + i_conn0, n_block_conn, n_source_tot);
      DBGCUDASYNC;
    }

    //conn_source_ids_offset += n_block_conn; uncomment only if needed

    // set the target indexes in the new connections using the fixed-indegree rule
    setIndegreeTarget< T2, ConnStructT > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>(
      conn_struct_vect_[ ib ] + i_conn0, n_block_conn, n_prev_conn, target, indegree );
    DBGCUDASYNC;
  } // end of loop on connection blocks
  
  int64_t i_conn0 = n_conn_ % conn_block_size_; // index of the first new connection
  
  // Sort the connections in the blocks with the COPASS algorithm
  // Sorting should start from block ib0 and it should be performed on
  // n_new_conn_tot + i_conn0 elements skipping the first i_conn0 connections 
    
  // Allocating auxiliary GPU memory
  int64_t sort_storage_bytes = 0;
  void* d_sort_storage = NULL;
  copass_sort::sort< ConnKeyT, ConnStructT >
    (&conn_key_vect_[ib0], &conn_struct_vect_[ib0], n_new_conn_tot + i_conn0,
     conn_block_size_, d_sort_storage, sort_storage_bytes, i_conn0 );
  // printf( "storage bytes: %ld\n", sort_storage_bytes );
  CUDAMALLOCCTRL( "&d_sort_storage", &d_sort_storage, sort_storage_bytes );

  // printf( "Sorting...\n" );
  copass_sort::sort< ConnKeyT, ConnStructT >
    (&conn_key_vect_[ib0], &conn_struct_vect_[ib0], n_new_conn_tot + i_conn0,
     conn_block_size_, d_sort_storage, sort_storage_bytes, i_conn0 );
  CUDAFREECTRL( "d_sort_storage", d_sort_storage );

  ///////////////////////////////////////////////////////////////////
  // Partition the new connections according to the source host
  // using the cumulative sum array n_source_cumul[n_source_host + 1] created previously
  // Each element of this array, n_source_cumul[i_host], 
  // represents the first index of the source nodes belonging to the host i_host in the host&node index
  // representation as integers from 0 to n_source_tot
  // Since the connection are sorted by the source host&node index in this representation,
  // this index can be used to partition them according to the source host through a search (locate)
  
  int ib1 = ib0;
  int64_t part_conn0 = n_conn_; // index of the first connection of the first partition
  
  // Loop on source hosts (it's fine to do it with a loop, no need for parallelizing)
  for (int ish=0, ish<n_source_host; ish++) {
    int64_t h_position;
    // Loop on connection blocks where new connections are stored
    int ib;
    for ( ib = ib1; ib < new_n_block; ib++ ) {
      int64_t n_block_conn; // number of new connections in the current block of the loop
      int64_t i_conn0;      // index of first new connection in this block
      if ( new_n_block == ib0 + 1 ) // no new blocks were needed/allocated
      { // all connections are in the same block
	i_conn0 = n_conn_ % conn_block_size_;
	n_block_conn = n_new_conn_tot;
      }
      else if ( ib == ib0 ) // first block of the loop, cannot be the last (see above)
      { // first block
	i_conn0 = n_conn_ % conn_block_size_;
	n_block_conn = conn_block_size_ - i_conn0;
      }
      else if ( ib == new_n_block - 1 ) // last block of the loop, cannot be the first (see above)
      { // last block
	i_conn0 = 0;
	n_block_conn = ( n_conn_ + n_new_conn_tot - 1 ) % conn_block_size_ + 1;
      }
      else // block is neither the first nor the last of the loop
      {
	i_conn0 = 0;
	n_block_conn = conn_block_size_;
      }

      // Locate (search) the current element of the n_source_cumul array in the conn_key_vect_ block
      int64_t value = n_source_cumul[ish+1];
      // allocate a 64 bit integer to store the search result
      int64_t *d_position;
      CUDAMALLOCCTRL( "&d_position", &d_position, sizeof(int64_t) );
      
      if (sizeof(ConnKeyT)==8) { // 64 bit
	// perform search in an array of 64 bit unsigned integers
	// Find number of elements < val in a sorted array array[i+1]>=array[i]
	search_down< unsigned long long, 1024 > <<< 1, 1024 >>>
	  ((unsigned long long*)conn_key_vect_[ ib ] + i_conn0,
	   n_block_conn, value, d_position, 0);
	DBGCUDASYNC;
      }
      else {
	// perform search in an array of 32 bit unsigned integers
	// Find number of elements < val in a sorted array array[i+1]>=array[i]
	search_down< unsigned int, 1024 > <<< 1, 1024 >>>
	  ((unsigned int*)conn_key_vect_[ ib ] + i_conn0,
	   n_block_conn, value, d_position, 0);
	DBGCUDASYNC;
      }
      // Copy position from GPU to CPU memory
      gpuErrchk( cudaMemcpy( &h_position, d_position, sizeof( int64_t ), cudaMemcpyDeviceToHost ) );
      // check if found
      if (h_position < n_block_conn) {
	ib1 = ib; // next partition search should start from current block, where current partition ends
	h_position += i_conn0
	break;
      }
    }
    if (ib == new_n_block) {
      throw ngpu_exception( "Search error in partitioning new connections in _ConnectDistrubutedFixedIndegree" );
    }
    // compute index of the last connection of the current partition + 1
    int64_t part_conn1 = ib1 * conn_block_size_ + h_position;
    // compute number of connections in current partition
    n_new_conn = part_conn1 - part_conn0;

    // Now we must subtract n_source_cumul[ish] and convert the source node indexes to the ConnKeyT representation
    // in all connections of all blocks of the current partition
    // To do this, we need to loop on all blocks of the current partition
    
    // Loop on connection blocks where connections of the current partition are stored
    //int64_t conn_source_ids_offset = 0; uncomment only if needed
    int64_t n_prev_conn = 0;
    int ib0 = ( int ) ( part_conn0 / conn_block_size_ );
    int nb = ib1 - ib0 + 1;
    for ( int ib = ib0; ib < nb; ib++ ) {
      int64_t n_block_conn; // number of new connections in the current block of the loop
      int64_t i_conn0;      // index of first new connection in this block
      if ( nb == ib0 + 1 ) // no new blocks were needed/allocated
      { // all connections are in the same block
	i_conn0 = part_conn0 % conn_block_size_;
	n_block_conn = n_new_conn;
      }
      else if ( ib == ib0 ) // first block of the loop, cannot be the last (see above)
      { // first block
	i_conn0 = part_conn0 % conn_block_size_;
	n_block_conn = conn_block_size_ - i_conn0;
      }
      else if ( ib == nb - 1 ) // last block of the loop, cannot be the first (see above)
      { // last block
	i_conn0 = 0;
	n_block_conn = ( part_conn0 + n_new_conn - 1 ) % conn_block_size_ + 1;
      }
      else // block is neither the first nor the last of the loop
      {
	i_conn0 = 0;
	n_block_conn = conn_block_size_;
      }
      // Launch CUDA kernel that subtracts n_source_cumul[ish] and convert the source node indexes
      // to the ConnKeyT representation in all connections of the current block
      subtractSourceOffsetKernel< T1, ConnKeyT > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>
	(conn_key_vect_[ ib ] + i_conn0, n_block_conn, n_source_cumul[ish], d_source_arr[ish]); 
      DBGCUDASYNC;
    }

    // do a regular RemoteConnectSource with source host n. ish
    // using a new connection rule that uses already created connections
    // filled only with source node relative indexes and target node
    // indexes and fills them with weights, delays, syn_geoups, ports
    
    part_conn0 = part_conn1; // update index of the first connection of the next partition


  }

  freeNodeArrayFromDevice(d_target);
  for (int ish=0; ish<n_source_host; ish++) {
    freeNodeArrayFromDevice(d_source_arr[ish]);
  }
  
  return ret;
}
      
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build connections with fixed indegree rule for source neurons and target neurons distributed across
// MPI processes (hosts)
// Case with both source and target nodes contiguous, represented by starting index and number of nodes 
///////////////////////////////////////////////////////////////////////////////////////////////////////
int
NESTGPU::ConnectDistributedFixedIndegree
(int *source_host_arr, int n_source_host, inode_t *source_arr, int *n_source_arr,
 int *target_host_arr, int n_target_host, inode_t *target_arr, int *n_target_arr,
 int indegree, int i_host_group, SynSpec &syn_spec)
{
  CheckUncalibrated( "Connections cannot be created after calibration" );
  int ret = conn_->connectDistributedFixedIndegree<inode_t, inode_t>
    (source_host_arr, n_source_host, source_arr, n_source_arr,
     target_host_arr, n_target_host, target_arr, n_target_arr, indegree, i_host_group, syn_spec);

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build connections with fixed indegree rule for source neurons and target neurons distributed across
// MPI processes (hosts)
// Case with source nodes stored in an array,
// target nodes contiguous, represented by starting index and number of nodes 
///////////////////////////////////////////////////////////////////////////////////////////////////////
int
NESTGPU::ConnectDistributedFixedIndegree
(int *source_host_arr, int n_source_host, inode_t **source_arr, int *n_source_arr,
 int *target_host_arr, int n_target_host, inode_t *target_arr, int *n_target_arr,
 int indegree, int i_host_group, SynSpec &syn_spec)
{
  CheckUncalibrated( "Connections cannot be created after calibration" );
  int ret = conn_->connectDistributedFixedIndegree<inode_t*, inode_t>
    (source_host_arr, n_source_host, source_arr, n_source_arr,
     target_host_arr, n_target_host, target_arr, n_target_arr, indegree, i_host_group, syn_spec);

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build connections with fixed indegree rule for source neurons and target neurons distributed across
// MPI processes (hosts)
// Case with source nodes contiguous, represented by starting index and number of nodes,
// target nodes stored in an array
///////////////////////////////////////////////////////////////////////////////////////////////////////
int
NESTGPU::ConnectDistributedFixedIndegree
(int *source_host_arr, int n_source_host, inode_t *source_arr, int *n_source_arr,
 int *target_host_arr, int n_target_host, inode_t **target_arr, int *n_target_arr,
 int indegree, int i_host_group, SynSpec &syn_spec)
{
  CheckUncalibrated( "Connections cannot be created after calibration" );
  int ret = conn_->connectDistributedFixedIndegree<inode_t*, inode_t>
    (source_host_arr, n_source_host, source_arr, n_source_arr,
     target_host_arr, n_target_host, target_arr, n_target_arr, indegree, i_host_group, syn_spec);

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build connections with fixed indegree rule for source neurons and target neurons distributed across
// MPI processes (hosts)
// Case with both source nodes and target nodes stored in arrays
///////////////////////////////////////////////////////////////////////////////////////////////////////
int
NESTGPU::ConnectDistributedFixedIndegree
(int *source_host_arr, int n_source_host, inode_t *source_arr, int *n_source_arr,
 int *target_host_arr, int n_target_host, inode_t **target_arr, int *n_target_arr,
 int indegree, int i_host_group, SynSpec &syn_spec)
{
  CheckUncalibrated( "Connections cannot be created after calibration" );
  int ret = conn_->connectDistributedFixedIndegree<inode_t*, inode_t>
    (source_host_arr, n_source_host, source_arr, n_source_arr,
     target_host_arr, n_target_host, target_arr, n_target_arr, indegree, i_host_group, syn_spec);

  return ret;
}

    
  
