#include <gtest/gtest.h>

// system
#include <limits>

#include "mtrand.h"

using namespace std;


TEST(mtrand, TestSeedMTrand_seed0_unlimited)
{
   MTRandom rng;
   int rngSeed = 0;
   rng.seed(rngSeed);
   EXPECT_EQ(rng.rand(), 2357136044); // 1 (seed 0)
   EXPECT_EQ(rng.rand(), 2546248239); // 2 (seed 0)
   EXPECT_EQ(rng.rand(), 3071714933); // 3 (seed 0)
   EXPECT_EQ(rng.rand(), 3626093760); // 4 (seed 0)
   EXPECT_EQ(rng.rand(), 2588848963); // 5 (seed 0)
}

TEST(mtrand, TestSeedMTrand_seed0_limit10)
{
   MTRandom rng;
   int rngSeed = 0;
   rng.seed(rngSeed);
   EXPECT_EQ(rng.rand()%10, 4); // 1 (seed 0)
   EXPECT_EQ(rng.rand()%10, 9); // 2 (seed 0)
   EXPECT_EQ(rng.rand()%10, 3); // 3 (seed 0)
   EXPECT_EQ(rng.rand()%10, 0); // 4 (seed 0)
   EXPECT_EQ(rng.rand()%10, 3); // 5 (seed 0)
   // repeat seed 0
   rng.seed(rngSeed);
   EXPECT_EQ(rng.rand()%10, 4); // 1 (seed 0)
}

TEST(mtrand, TestSeedMTrand_seed1_limit10)
{
   MTRandom rng;
   int rngSeed = 1;
   rng.seed(rngSeed);
   EXPECT_EQ(rng.rand()%10, 5); // 1 (seed 0)
   EXPECT_EQ(rng.rand()%10, 9); // 2 (seed 0)
   EXPECT_EQ(rng.rand()%10, 4); // 3 (seed 0)
   EXPECT_EQ(rng.rand()%10, 8); // 4 (seed 0)
   EXPECT_EQ(rng.rand()%10, 3); // 5 (seed 0)
}
