#ifndef SUMMATION_H
#define SUMMATION_H

/*!
	Kahan summation
*/
template< class WORK_TYPE>
WORK_TYPE summation_k(WORK_TYPE a, WORK_TYPE b, WORK_TYPE* correction)
{
	WORK_TYPE	corrected = b - *correction;
	WORK_TYPE	new_sum = a + corrected;
	*correction = (new_sum - a) - corrected;
	return new_sum;
}

/*!
	Kahan array range summation
*/
template< class WORK_TYPE, class A >
WORK_TYPE summation_k(A container, size_t begin, size_t end, WORK_TYPE* correction)
{
	if(begin == end)
	{
		return WORK_TYPE(0);
	}
	WORK_TYPE sum;

	sum = container[begin];
	for(size_t i = begin + 1; i < end; ++i)
	{
		sum = summation_k(sum, container[i], correction);
	}
	return sum;
}

/*!
	Kahan array summation
*/
template< class WORK_TYPE, class A >
WORK_TYPE summation(const A& container, size_t size)
{
	WORK_TYPE correction(0);

	return summation_k(container, 0, size, &correction);
}

#endif // SUMMATION_H
