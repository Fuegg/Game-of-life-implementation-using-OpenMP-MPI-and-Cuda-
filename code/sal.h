#ifndef DUMMY_SAL_H
#define DUMMY_SAL_H

// On définit des macros vides pour tromper le fichier mpi.h de Microsoft
#define _In_
#define _Out_
#define _Inout_
#define _In_opt_
#define _Out_opt_
#define _Inout_opt_
#define _In_reads_(s)
#define _Out_writes_(s)
#define _Inout_updates_(s)
#define _In_reads_opt_(s)
#define _Out_writes_opt_(s)
#define _Out_writes_to_(s, c)
#define _In_reads_bytes_(s)
#define _Out_writes_bytes_(s)
#define _Out_writes_bytes_to_(s, c)
#define _Pre_satisfies_(e)
#define _Success_(e)

#endif