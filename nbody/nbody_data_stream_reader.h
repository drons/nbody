#ifndef NBODY_DATA_STREAM_READER_H
#define NBODY_DATA_STREAM_READER_H

#include "nbtype.h"

class nbody_data;

class nbody_data_stream_reader
{
	struct	data;
	data*	d;

	nbody_data_stream_reader(const nbody_data_stream_reader&);
	nbody_data_stream_reader& operator = (const nbody_data_stream_reader&);
public:
	nbody_data_stream_reader();
	virtual ~nbody_data_stream_reader();

	/*!
	   \param file_base_name - stream base name to open
	   \return 0 on success
	 */
	int load(const QString& file_base_name);

	/*!
	   \brief close data stream
	 */
	void close();

	/*!
	   \return frame count in stream
	 */
	size_t get_frame_count() const;

	/*!
	   \return ODE solve steps count in stream
	 */
	size_t get_steps_count() const;

	/*!
	   \return stream duration
	 */
	nbcoord_t get_max_time() const;

	/*!
	   \param frame - new frame number to seek
	   \return 0 on success
	 */
	int seek(size_t frame);

	/*!
	   \return get current frame position
	*/
	size_t get_current_frame() const;

	/*!
	   \return get body count
	*/
	size_t get_body_count() const;

	/*!
	   \return get bytes count per coord value
	*/
	size_t get_coord_size() const;

	/*!
	   \return data box size
	*/
	nbcoord_t get_box_size() const;

	/*!
	   \brief read stream data to nbody_engine
	   \param e - destination engine
	   \return 0 on success
	 */
	int read(nbody_data* bdata);
};

#endif // NBODY_DATA_STREAM_READER_H
