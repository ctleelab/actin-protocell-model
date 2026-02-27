// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef CAPPING_FIBER_H
#define CAPPING_FIBER_H

#include "sim.h"
#include "vector.h"
#include "node_list.h"
#include "fiber.h"

class CappingFiberProp;


/// A Fiber with constant growth/shrinkage at both ends. With capping
/**
 This fiber grows and shrinks like the treadmilling fiber but can also be capped!

 See the @ref CappingFiberPar.

 @todo Document CappingFiber
 @ingroup FiberGroup
 */
class CappingFiber : public Fiber
{   
private:
    
    /// State of fiber ends
    state_t     mState[2];

    /// Assembly during last time-step
    real        mGrowth[2];

    /// Gillespie countdown timers for capping:
    real       nextCap[2];

    /// Gillespie countdown timers for uncapping: 
    real       nextUncap[2];
    
    /// Cache for fiber state before capping
    state_t    stateCache[2];

    /// Capping state
    bool       capped[2];

public:
    
    /// the Property of this object
    CappingFiberProp const* prop;
  
    /// constructor
    CappingFiber(CappingFiberProp const*);

    /// destructor
    virtual ~CappingFiber();
        
    //--------------------------------------------------------------------------
    
    /// return assembly/disassembly state of MINUS_END
    state_t     endStateM() const;
    
    /// change state of MINUS_END
    void        setEndStateM(state_t s);
    
    /// the amount of freshly assembled polymer during the last time step
    real        freshAssemblyM() const;

    
    /// return assembly/disassembly state of PLUS_END
    state_t     endStateP() const;

    /// change state of PLUS_END
    void        setEndStateP(state_t s);
    
    /// the amount of freshly assembled polymer during the last time step
    real        freshAssemblyP() const;
    
    //--------------------------------------------------------------------------
   
    /// check capping
    void        updateCapping();

    /// monte-carlo step
    void        step();
    
    //--------------------------------------------------------------------------
    
    /// write to Outputter
    void        write(Outputter&) const;

    /// read from Inputter
    void        read(Inputter&, Simul&, ObjectTag);
    
};


#endif
