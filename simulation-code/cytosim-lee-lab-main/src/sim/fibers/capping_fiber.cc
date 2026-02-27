// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "dim.h"
#include "assert_macro.h"
#include "capping_fiber.h"
#include "capping_fiber_prop.h"
#include "exceptions.h"
#include "iowrapper.h"
#include "simul.h"
#include "space.h"


//------------------------------------------------------------------------------

CappingFiber::CappingFiber(CappingFiberProp const* p) : Fiber(p), prop(p)
{
    for (std::size_t i = 0; i < 2; ++i){
        mState[i] = STATE_WHITE;
        mGrowth[i] = 0;
        nextCap[i] = RNG.exponential();
        nextUncap[i] = RNG.exponential();
        capped[i]  = p->capped[i]; 
        stateCache[i] = STATE_GREEN;
    }
}


CappingFiber::~CappingFiber()
{
    prop = nullptr;
}


//------------------------------------------------------------------------------
#pragma mark -

state_t CappingFiber::endStateM() const
{
    return mState[1];
}


void CappingFiber::setEndStateM(state_t s)
{
    if ( s == STATE_WHITE || s == STATE_GREEN || s == STATE_YELLOW || s == STATE_RED )
        mState[1] = s;
    else
        throw InvalidParameter("Invalid AssemblyState for CappingFiber MINUS_END");
}


real CappingFiber::freshAssemblyM() const
{
    return mGrowth[1];
}


state_t CappingFiber::endStateP() const
{
    return mState[0];
}


void CappingFiber::setEndStateP(state_t s)
{
    if ( s == STATE_WHITE || s == STATE_GREEN || s == STATE_YELLOW || s == STATE_RED )
        mState[0] = s;
    else
        throw InvalidParameter("Invalid AssemblyState for CappingFiber PLUS_END");
}


real CappingFiber::freshAssemblyP() const
{
    return mGrowth[0];
}


//------------------------------------------------------------------------------


void CappingFiber::updateCapping(){
    for(std::size_t i = 0; i < 2; ++i){
        // obscure ternary operator maps 0 to PLUS_END and 1 to MINUS_END
        FiberEnd end = i ? MINUS_END : PLUS_END;
        if ( capped[i] ){
            if ( endState(end) != STATE_YELLOW){
                stateCache[i] = endState(end); 
                setEndState(end, STATE_YELLOW);
            }
            nextUncap[i] -= prop->uncapping_rate_dt[i];
            // uncapping condition
            if ( nextUncap[i] < 0){
                capped[i] = false; // uncap
                setEndState(end, stateCache[i]);
                nextCap[i] = RNG.exponential();
            } 
        }
        else {
            nextCap[i] -= prop->capping_rate_dt[i];
            // capping condition
            if ( nextCap[i] < 0 ){
                stateCache[i] = endState(end);
                setEndState(end, STATE_YELLOW);
                capped[i] = true; // cap
                nextUncap[i] = RNG.exponential();
            }
        }
    }
}

void CappingFiber::step()
{   
    // First update capping states
    updateCapping();

    for(std::size_t i = 0; i < 2; ++i){
        // obscure ternary operator maps 0 to PLUS_END and 1 to MINUS_END
        FiberEnd end = i ? MINUS_END : PLUS_END;
    
        if ( capped[i] ){
            mGrowth[i] = 0;
        } 
        else {
            if ( mState[i] == STATE_GREEN )
            {
                // calculate the force acting on the point at the end:
                real force = projectedForceEnd(end);
                
                // growth is reduced if free monomers are scarce:
                mGrowth[i] = prop->growing_speed_dt[i] * prop->free_polymer;
                
                assert_true(mGrowth[i] >= 0);

                // antagonistic force (< 0) decreases assembly rate exponentially
                if ( force < 0  &&  prop->growing_force[i] < INFINITY )
                    mGrowth[i] *= exp(force/prop->growing_force[i]);
            }
            else if ( mState[i] == STATE_RED )
            {
                mGrowth[i] = prop->shrinking_speed_dt[i];
            }
            else
            {
                mGrowth[i] = 0;
            }
        }

    }


    real len = length();
    real inc = mGrowth[0] + mGrowth[1];
    if ( inc < 0  &&  len + inc < prop->min_length )
    {
        // the fiber is too short, we delete it:
        delete(this);
        return;
    }
    else if ( len + inc < prop->max_length )
    {
        if ( mGrowth[1] ) growM(mGrowth[1]);
        if ( mGrowth[0] ) growP(mGrowth[0]);
    }
    else if ( len < prop->max_length )
    {
        // the remaining possible growth is distributed to the two ends:
        inc = ( prop->max_length - len ) / inc;
        if ( mGrowth[1] ) growM(inc*mGrowth[1]);
        if ( mGrowth[0] ) growP(inc*mGrowth[0]);
    }

    Fiber::step();
}

                  
//------------------------------------------------------------------------------
#pragma mark -


void CappingFiber::write(Outputter& out) const
{
    Fiber::write(out);

    /// write variables describing the dynamic state of the ends:
    writeMarker(out, DYNAMIC_TAG);
    for (std::size_t i = 0; i < 2; ++i){
        out.writeUInt16(mState[i]);
        out.writeUInt16(capped[i]);
    }
}


void CappingFiber::read(Inputter& in, Simul& sim, ObjectTag tag)
{
#ifdef BACKWARD_COMPATIBILITY
    if ( tag == DYNAMIC_TAG || ( tag == TAG && in.formatID() < 44 ) )
#else
    if ( tag == TAG_DYNAMIC )
#endif
    {
        for (std::size_t i = 0; i < 2; ++i){
            mState[i] = in.readUInt16();
            capped[i] = in.readUInt16(); 
        }
    }
#ifdef BACKWARD_COMPATIBILITY
    if ( tag != DYNAMIC_TAG || in.formatID() < 44 )
#else
    else
#endif
        Fiber::read(in, sim, tag);
}
