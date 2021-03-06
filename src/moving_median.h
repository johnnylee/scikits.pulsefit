
#include <Python.h>
#include <numpy/arrayobject.h>

#include <errno.h>
#include <math.h>

// This is the branching factor for the heap.
#define NUM_CHILDREN 8

// Minimum of two numbers.
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Find indices of parent and the first child.
#define P_IDX(i) ((i) - 1) / NUM_CHILDREN
#define FC_IDX(i) NUM_CHILDREN * (i) + 1

/*****************************************************************************
 * mm_node represents a node in the heap.
 *****************************************************************************/
struct _mm_node {
  int             small; // 1 if the node is in the small heap.
  npy_int64       idx;   // The node's index in the heap array.
  npy_float64     val;   // The node's value.
  struct _mm_node *next; // The next node in order of insertion.
};
typedef struct _mm_node mm_node;


/*****************************************************************************
 * mm_handle is the double-heap "class". 
 *****************************************************************************/
struct _mm_handle {
  int       odd;       // 1 if the window size is odd, 0 otherwise.
  npy_int64 n_s;       // The number of elements in the min heap.
  npy_int64 n_l;       // The number of elements in the max heap. 

  mm_node **s_heap;    // The min heap.
  mm_node **l_heap;    // The max heap.
  mm_node **nodes;     // All the nodes. s_heap and l_heap point into
                       // this array.
  mm_node  *node_data; // Pointer to memory location where nodes live. 
  mm_node  *first;     // The node added first to the list of nodes. 
  mm_node  *last;      // The last (most recent) node added. 
  
  // Most nodes are leaf nodes, therefore it makes sense to have a
  // quick way to check if a node is a leaf to avoid processing.
  npy_int64 s_first_leaf; // First leaf index in the small heap.
  npy_int64 l_first_leaf; // First leaf index in the large heap.
};
typedef struct _mm_handle mm_handle;


/*****************************************************************************
 * Internal functions are defined in this section. 
 *****************************************************************************/

/*
 * Return the index of the smallest child of the node. The pointer
 * child will also be set.
 */
inline npy_int64 get_smallest_child(mm_node   **heap,
                                    npy_int64 size,
                                    npy_int64 idx,
                                    mm_node   **child) {
  npy_int64 i0 = FC_IDX(idx);
  npy_int64 i1 = i0 + NUM_CHILDREN;
  i1 = min(i1, size);
  
  switch(i1 - i0) {
  case  8: if(heap[i0 + 7]->val < heap[idx]->val) { idx = i0 + 7; }
  case  7: if(heap[i0 + 6]->val < heap[idx]->val) { idx = i0 + 6; }
  case  6: if(heap[i0 + 5]->val < heap[idx]->val) { idx = i0 + 5; }
  case  5: if(heap[i0 + 4]->val < heap[idx]->val) { idx = i0 + 4; }
  case  4: if(heap[i0 + 3]->val < heap[idx]->val) { idx = i0 + 3; }
  case  3: if(heap[i0 + 2]->val < heap[idx]->val) { idx = i0 + 2; }
  case  2: if(heap[i0 + 1]->val < heap[idx]->val) { idx = i0 + 1; }
  case  1: if(heap[i0    ]->val < heap[idx]->val) { idx = i0;     }
  }
  
  *child = heap[idx];
  return idx;
}

/*
 * Return the index of the largest child of the node. The pointer
 * child will also be set. 
 */
inline npy_int64 get_largest_child(mm_node **heap,
                                   npy_int64   size,
                                   npy_int64   idx,
                                   mm_node  **child) {
  npy_int64 i0 = FC_IDX(idx);
  npy_int64 i1 = i0 + NUM_CHILDREN;
  i1 = min(i1, size);
  
  switch(i1 - i0) {
  case  8: if(heap[i0 + 7]->val > heap[idx]->val) { idx = i0 + 7; }
  case  7: if(heap[i0 + 6]->val > heap[idx]->val) { idx = i0 + 6; }
  case  6: if(heap[i0 + 5]->val > heap[idx]->val) { idx = i0 + 5; }
  case  5: if(heap[i0 + 4]->val > heap[idx]->val) { idx = i0 + 4; }
  case  4: if(heap[i0 + 3]->val > heap[idx]->val) { idx = i0 + 3; }
  case  3: if(heap[i0 + 2]->val > heap[idx]->val) { idx = i0 + 2; }
  case  2: if(heap[i0 + 1]->val > heap[idx]->val) { idx = i0 + 1; }
  case  1: if(heap[i0    ]->val > heap[idx]->val) { idx = i0;     }
  }

  *child = heap[idx];
  return idx;
}

/*
 * Swap nodes.
 */
#define SWAP_NODES(heap, idx1, node1, idx2, node2) \
  heap[idx1] = node2;                              \
  heap[idx2] = node1;                              \
  node1->idx = idx2;                               \
  node2->idx = idx1;                               \
  idx1       = idx2

/*
 * Move the given node up through the heap to the appropriate position. 
 */
inline void move_up_small(mm_node   **heap,
                          npy_int64 idx,
                          mm_node   *node,
                          npy_int64 p_idx,
                          mm_node   *parent) {
  do {
    SWAP_NODES(heap, idx, node, p_idx, parent);
    if(idx == 0) {
      break;
    }
    p_idx = P_IDX(idx);
    parent = heap[p_idx];
  } while (node->val > parent->val);
}

/*
 * Move the given node down through the heap to the appropriate position. 
 */ 
inline void move_down_small(mm_node   **heap,
                            npy_int64 size,
                            npy_int64 idx,
                            mm_node   *node) {
  mm_node *child;
  npy_float64 val = node->val;
  npy_int64 c_idx = get_largest_child(heap, size, idx, &child);

  while(val < child->val) {
    SWAP_NODES(heap, idx, node, c_idx, child);
    c_idx = get_largest_child(heap, size, idx, &child);
  }
}

/*
 * Move the given node down through the heap to the appropriate
 * position.
 */ 
inline void move_down_large(mm_node   **heap,
                            npy_int64 idx,
                            mm_node   *node,
                            npy_int64 p_idx,
                            mm_node   *parent) {
  do {
    SWAP_NODES(heap, idx, node, p_idx, parent);
    if(idx == 0) {
      break;
    }
    p_idx = P_IDX(idx);
    parent = heap[p_idx];
  } while (node->val < parent->val);
}

/*
 * Move the given node up through the heap to the appropriate position. 
 */ 
inline void move_up_large(mm_node   **heap,
                          npy_int64 size,
                          npy_int64 idx,
                          mm_node   *node) {
  mm_node *child;
  npy_float64 val   = node->val;
  npy_int64 c_idx = get_smallest_child(heap, size, idx, &child);

  while(val > child->val) {
    SWAP_NODES(heap, idx, node, c_idx, child);
    c_idx = get_smallest_child(heap, size, idx, &child);
  }
}

/*
 * Swap the heap heads. 
 */
inline void swap_heap_heads(mm_node   **s_heap,
                            npy_int64 n_s,
                            mm_node   **l_heap,
                            npy_int64 n_l,
                            mm_node   *s_node,
                            mm_node   *l_node) {
  s_node->small = 0;
  l_node->small = 1;
  s_heap[0] = l_node;
  l_heap[0] = s_node;
  move_down_small(s_heap, n_s, 0, l_node);
  move_up_large(l_heap, n_l, 0, s_node);
}


/*****************************************************************************
 * Public functions are defined below. 
 *****************************************************************************/

/* 
 * Construct the double heap with the given total number of values,
 * `size`. Return the mm_handle structure, uninitialized.
 */
mm_handle *mm_new(npy_int64 size) {
  mm_handle *mm = malloc(sizeof(mm_handle));
  mm->nodes = malloc(size * sizeof(mm_node*));
  mm->node_data = malloc(size * sizeof(mm_node));

  if(mm == NULL || mm->nodes == NULL || mm->node_data == NULL) {
    printf("Malloc failure.\n");
    printf("Sizes: %i, %i, %i\n", 
           (int)sizeof(mm_handle), 
           (int)(size * sizeof(mm_node*)),
           (int)(size * sizeof(mm_node)));
    printf("Error: %i\n", errno);
    printf("%s", strerror(errno));
    printf("\n");
    exit(-1);
  }

  mm->odd = size % 2;
  mm->n_l = 0;
  mm->n_s = 0;

  mm->s_heap = mm->nodes;
  mm->l_heap = &mm->nodes[size/2 + size % 2];
  
  return mm;
}

/*
 * Update the running median with a new value. 
 */
void mm_update(mm_handle *mm, npy_float64 val) {

  // Local variables.
  mm_node **l_heap = mm->l_heap;
  mm_node **s_heap = mm->s_heap;
  npy_int64 n_s    = mm->n_s;
  npy_int64 n_l    = mm->n_l;

  // Nodes and indices. 
  mm_node *node = mm->first;
  npy_int64 idx = node->idx;

  mm_node *node2;
  npy_int64 idx2;

  // Replace value of first inserted node, and update first, last.
  node->val = val;
  mm->first = mm->first->next; 
  mm->last->next = node;
  mm->last = node;

  // In small heap.
  if(node->small) {

    // Internal or leaf node. 
    if(idx > 0) {
      idx2 = P_IDX(idx);
      node2 = s_heap[idx2];
      
      // Move up.
      if(val > node2->val) {
        move_up_small(s_heap, idx, node, idx2, node2);
        
        // Maybe swap between heaps.
        node2 = l_heap[0];
        if(val > node2->val) {
          swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
        }
      }

      // Move down. 
      else if(idx < mm->s_first_leaf) {
        move_down_small(s_heap, n_s, idx, node);
      }
    }
    
    // Head node. 
    else {
      node2 = l_heap[0];
      if(val > node2->val) {
        swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
      } else {
        move_down_small(s_heap, n_s, idx, node);
      }
    }
  }

  // In large heap. 
  else {

    // Internal or leaf node. 
    if(idx > 0) {
      idx2 = P_IDX(idx);
      node2 = l_heap[idx2];
      
      // Move down.
      if(val < node2->val) {
        move_down_large(l_heap, idx, node, idx2, node2);
        
        // Maybe swap between heaps.
        node2 = s_heap[0];
        if(val < node2->val) {
          swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
        }
      }

      // Move up. 
      else if(idx < mm->l_first_leaf) {
        move_up_large(l_heap, n_l, idx, node);
      }
    }
    
    // Head node. 
    else {
      node2 = s_heap[0];
      if(val < node2->val) {
        swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
      } else {
        move_up_large(l_heap, n_l, idx, node);
      }
    }
  }
}


/*
 * Insert initial values into the double heap structure. This should
 * be called a number of times equal to the size of the heap before
 * calling mm_get_median.
 */
void mm_insert_init(mm_handle *mm, npy_float64 val) {
  // Some local variables. 
  mm_node *node;
  npy_int64 n_s = mm->n_s;
  npy_int64 n_l = mm->n_l;


  // The first node. 
  if(n_s == 0) {
    node = mm->s_heap[0] = &mm->node_data[0];
    node->small = 1;
    node->idx   = 0;
    node->val   = val;
    node->next  = mm->l_heap[0];

    mm->n_s = 1;
    mm->first = mm->last = node;
    mm->s_first_leaf = 0;
  } 
  
  // Nodes after the first. 
  else {

    // Add to the large heap. 
    if(n_s > n_l) {
      node = mm->l_heap[n_l] = &mm->node_data[n_s + n_l];
      node->small = 0;
      node->idx   = n_l;
      node->next  = mm->first;

      mm->first = node;
      ++mm->n_l;
      mm->l_first_leaf = ceil((mm->n_l - 1) / (double)NUM_CHILDREN);
      mm_update(mm, val);
    } 
    
    // Add to the small heap.
    else {
      node = mm->s_heap[n_s] = &mm->node_data[n_s + n_l];
      node->small = 1;
      node->idx   = n_s;
      node->next  = mm->first;

      mm->first = node;
      ++mm->n_s;
      mm->s_first_leaf = ceil((mm->n_s - 1) / (double)NUM_CHILDREN);
      mm_update(mm, val);
    }
  }
}

/*
 * Return the current median value. 
 */
npy_float64 mm_get_median(mm_handle *mm) {
  if(mm->odd) {
    return mm->s_heap[0]->val;
  } else {
    return (mm->s_heap[0]->val + mm->l_heap[0]->val) / 2.0;
  }
}

/*
 * Print the two heaps to the screen.
 */
void mm_dump(mm_handle *mm) {
  printf("\n\nFirst: %f\n", (double)mm->first->val);
  printf("Last: %f\n", (double)mm->last->val);
  
  npy_int64 i;
  printf("\n\nSmall heap: %i\n", (int)mm->n_s);
  for(i = 0; i < mm->n_s; ++i) {
    printf("%i %f\n", (int)mm->s_heap[i]->idx, mm->s_heap[i]->val);
    }
  
  printf("\n\nLarge heap: %i\n", (int)mm->n_l);
  
  for(i = 0; i < mm->n_l; ++i) {
    printf("%i %f\n", (int)mm->l_heap[i]->idx, mm->l_heap[i]->val);
    }
}

/*
 * Free memory allocated in mm_new.
 */
void mm_free(mm_handle *mm) {
  free(mm->node_data);
  free(mm->nodes);
  free(mm);
}
